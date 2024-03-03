import copy

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd

# torch.backends.cuda.matmul.allow_tf32 = True


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype)  # dummy loss value

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class MyStableDiffusion(nn.Module):
    def __init__(self, device, model_key, opt=None):
        super().__init__()

        self.device = device
        self.opt = opt
        self.register_store = {'se_step': None, 'skip_feature': None, 'mid_feature': None, 'lora_scale': None,
                               'use_parallel': False, 'timesteps': [], 'bs': opt.batch_size}

        print(f'[INFO] loading stable diffusion...')

        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", torch_dtype=opt.dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer", torch_dtype=opt.dtype)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", torch_dtype=opt.dtype)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", torch_dtype=opt.dtype)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)


        self.vae.to(device)  # 1151MB
        self.text_encoder.to(device)  # 1296MB
        self.unet.to(device)  # 3366MB

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        if opt.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * opt.t_range[0])
        self.max_step = int(self.num_train_timesteps * opt.t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(opt.dtype).to(self.device)  # for convenience

        # unconditional embeddings
        self.negative_prompts = ['ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers'] \
                                * opt.batch_size
        with torch.no_grad():
            uncond_input = self.tokenizer(self.negative_prompts, padding='max_length',
                                          max_length=self.tokenizer.model_max_length, return_tensors='pt')
            self.uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]  # 382MB

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        if not self.opt.use_embeddings:
            # Tokenize text and get embeddings
            text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                        truncation=True, return_tensors='pt')

            with torch.no_grad():
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

            # Do the same for unconditional embeddings
            uncond_input = self.tokenizer(negative_prompt, padding='max_length',
                                          max_length=self.tokenizer.model_max_length, return_tensors='pt')

            with torch.no_grad():
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        else:
            with torch.no_grad():
                uncond_embeddings = self.uncond_embeddings.to(self.device)

                text_embeddings = torch.cat([torch.load(pth).to(self.device) for pth in self.pth_path])

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                        latents=None, teacher='T2ITeacher'):

        if latents is None:
            latents = torch.randn(
                (text_embeddings.shape[0] // 2, self.unet.config.in_channels, height // 8, width // 8),
                device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    assert teacher in ['T2ITeacher', 'LoRATeacher']
                    if teacher == 'T2ITeacher':
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
                    elif teacher == 'LoRATeacher':
                        noise_pred = self.lora_unet(latent_model_input, t, encoder_hidden_states=text_embeddings)[
                            'sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    # @torch.no_grad()
    def decode_latents(self, latents):
        latents.to(self.vae.device)

        latents = 1 / 0.18215 * latents

        imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompts_to_latents(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,
                      guidance_scale=7.5, latents=None, teacher='T2ITeacher'):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(prompts)

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                                       teacher=teacher)  # [1, 4, 64, 64]

        # # Img latents -> imgs
        # imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        #
        # # Img to Numpy
        # imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        # imgs = (imgs * 255).round().astype('uint8')

        return latents

    def produce_latents_student(self, student_unet, text_embeddings, height=512, width=512, num_inference_steps=4,
                                latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0], 4, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        if self.register_store['use_parallel']:
            self.register_store['timesteps'] = self.scheduler.timesteps

        for i, t in enumerate(self.scheduler.timesteps):
            # predict the noise residual
            self.register_store['se_step'] = i
            noise_pred = student_unet(latents, t, encoder_hidden_states=text_embeddings)['sample']

            if self.register_store['use_parallel']:
                break

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        if self.register_store['use_parallel']:
            for i, t in enumerate(self.register_store['timesteps']):
                curr_noise = noise_pred[i * self.opt.batch_size: (i + 1) * self.opt.batch_size]
                latents = self.scheduler.step(curr_noise, t, latents)['prev_sample']

        return latents

    def prompt_to_img_student(self, student_unet, prompts, negative_prompts='', height=512, width=512,
                              num_inference_steps=4, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(prompts)

        if negative_prompts == [''] * len(prompts):
            negative_prompts = self.negative_prompts

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        _, text_embeddings = text_embeds.chunk(2)

        # Text embeds -> img latents
        latents = self.produce_latents_student(student_unet, text_embeddings, height=height, width=width,
                                               num_inference_steps=num_inference_steps,
                                               latents=latents)  # [1, 4, 64, 64]

        # with torch.no_grad():
        #     # Img latents -> imgs
        #     imgs = self.decode_latents(latents)  # [1, 3, 512, 512]
        #
        #     # Img to Numpy
        #     from PIL import Image
        #     imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        #     imgs = (imgs * 255).astype('uint8')
        #     imgs = self.image_grid(imgs, grid_size=(imgs.shape[0], 1))
        #     imgs = Image.fromarray(imgs)
        #     imgs.save(f'student.jpg')

        return latents, text_embeds

    def image_grid(self, img, grid_size):
        gw, gh = grid_size
        _N, H, W, C = img.shape
        img = img.reshape(gh, gw, H, W, C)
        img = img.transpose(0, 2, 1, 3, 4)
        img = img.reshape(gh * H, gw * W, C)
        return img



if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='A pineapple', type=str,)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--steps', type=int, default=50)


    # network backbone
    parser.add_argument('--sd_version', type=str, default='1.5', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--sd_base_path', type=str, default='/home/share/', choices=['/data/', '/home/share/',''], help="'/data/' for 3090, '/home/share/' for a40")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98], help="stable diffusion time steps range")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")

    parser.add_argument('--use_embeddings', type=bool, default=False, help="use_embeddings")
    parser.add_argument('--dtype', type=torch.dtype, default=torch.float32, help="dtype", choices=[torch.float16, torch.float32])
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool, default=True,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    opt = parser.parse_args()

    device = torch.device('cuda')

    sd = MyStableDiffusion(device, opt.sd_version, opt.hf_key, opt)

    imgs = sd.prompts_to_latents(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
