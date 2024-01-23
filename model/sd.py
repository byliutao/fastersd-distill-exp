import copy
import random

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, AttnProcessor
from diffusers.utils.import_utils import is_xformers_available
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd

# torch.backends.cuda.matmul.allow_tf32 = True

import wandb


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad) 
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype) # dummy loss value

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.1', hf_key=None, opt=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.opt = opt
        self.sd_base_path = self.opt.sd_base_path
        self.register_store = {'se_step': None, 'skip_feature': None, 'mid_feature': None, 'lora_scale': None,
                               'use_parallel': False, 'timesteps': [], 'bs': opt.batch_size}

        print(f'[INFO] loading stable diffusion...')
        
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = f"{self.sd_base_path}stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "/data/stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "/data/runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", torch_dtype=opt.dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer", torch_dtype=opt.dtype)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", torch_dtype=opt.dtype)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", torch_dtype=opt.dtype)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.lora_unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", torch_dtype=opt.dtype)
        self.lora_unet.requires_grad_(False)

        if not opt.lora:
            self.lora_unet.train()
            self.lora_unet.requires_grad_(True)
        else:
            lora_attn_procs = {}
            for name in self.lora_unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else self.lora_unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = self.lora_unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.lora_unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.lora_unet.config.block_out_channels[block_id]
                # LoRA rank=64 and alpha=108 following SwiftBrush
                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=64, network_alpha=108)
            self.lora_unet.set_attn_processor(lora_attn_procs)
            self.lora_layers = AttnProcsLayers(self.lora_unet.attn_processors)

        self.vae.to(device)  # 1151MB
        self.text_encoder.to(device)  # 1296MB
        self.unet.to(device)  # 3366MB
        self.lora_unet.to(device)  # 3402MB

        if opt.init_lora:
            self.init_lora_layers = copy.deepcopy(self.lora_layers)  # 52MB

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        if opt.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        if opt.use_wandb:
            wandb.watch(self.unet)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * opt.t_range[0])
        self.max_step = int(self.num_train_timesteps * opt.t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(opt.dtype).to(self.device) # for convenience

        # unconditional embeddings
        self.negative_prompts = ['ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers'] \
                                * opt.batch_size
        with torch.no_grad():
            uncond_input = self.tokenizer(self.negative_prompts, padding='max_length',  max_length=self.tokenizer.model_max_length, return_tensors='pt')
            self.uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]  # 382MB

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        if not self.opt.use_embeddings:
            # Tokenize text and get embeddings
            text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

            with torch.no_grad():
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

            # Do the same for unconditional embeddings
            uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

            with torch.no_grad():
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        else:
            with torch.no_grad():
                uncond_embeddings = self.uncond_embeddings.to(self.device)

                text_embeddings = torch.cat([torch.load(pth).to(self.device) for pth in self.pth_path])

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def train_step(self, text_embedding, latents_student, guidance_scale=7.5, lc_augment=False, grad_clip = None):

        # student
        assert torch.isnan(latents_student).sum() == 0, print(latents_student)
        latents = latents_student

        if lc_augment:  # https://github.com/mit-han-lab/data-efficient-gans
            from utils.DiffAugment_pytorch import DiffAugment
            policy = 'translation,cutout'
            latents = DiffAugment(latents, policy=policy)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [self.opt.batch_size], dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            latents_vsd = latents
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2)
            t_input = torch.cat([t] * 2)
            noise_pred = self.unet(latent_model_input, t_input, text_embedding).sample  # pred noise by T2I Teacher

            # perform guidance 7.5 following SwiftBrush for T2I Teacher
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.opt.sds is False:
                noise_pred_lora = self.lora_unet(latent_model_input, t_input, text_embedding).sample  # pred noise by LoRA Teacher

                # perform guidance 7.5 following SwiftBrush for LoRA Teacher
                noise_pred_lora_uncond, noise_pred_lora_text = noise_pred_lora.chunk(2)
                noise_pred_lora = noise_pred_lora_uncond + guidance_scale * (noise_pred_lora_text - noise_pred_lora_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])[:, None, None, None]
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        if self.opt.sds:  # sds loss
            grad = w * (noise_pred - noise)
        else:  # loss vsd
            grad = w * (noise_pred - noise_pred_lora.detach())

        # clip grad for stable training?
        if grad_clip is not None:
            grad = grad.clamp(-grad_clip, grad_clip)
        grad = torch.nan_to_num(grad)

        # dds loss: Delta Denoising Score (https://arxiv.org/pdf/2304.07090.pdf)
        # https://github.com/google/prompt-to-prompt/blob/main/DDS_zeroshot.ipynb
        if self.opt.loss_type == 'dds':
            loss = latents * grad.clone()
            loss = loss.sum() / (latents.shape[2] * latents.shape[3])
        # vsd loss: ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation (https://arxiv.org/pdf/2305.16213.pdf)
        # https://github.com/threestudio-project/threestudio/blob/main/threestudio/models/guidance/stable_diffusion_vsd_guidance.py#L652
        elif self.opt.loss_type == 'vsd':
            target = (latents_vsd - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents_vsd, target, reduction="mean")

        return loss, latents_student

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, teacher='T2ITeacher'):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.config.in_channels, height // 8, width // 8), device=self.device)

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
                        noise_pred = self.lora_unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):
        latents.to(self.vae.device)

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, teacher='T2ITeacher'):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(prompts)

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, teacher=teacher) # [1, 4, 64, 64]
        
        # # Img latents -> imgs
        # imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        #
        # # Img to Numpy
        # imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        # imgs = (imgs * 255).round().astype('uint8')

        return latents

    def produce_latents_student(self, student_unet, text_embeddings, height=512, width=512, num_inference_steps=4, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0], 4, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        if self.register_store['use_parallel']:
            self.register_store['timesteps'] = self.scheduler.timesteps

        self.noises_pred = []
        for i, t in enumerate(self.scheduler.timesteps):
            # predict the noise residual
            self.register_store['se_step'] = i
            noise_pred = student_unet(latents, t, encoder_hidden_states=text_embeddings)['sample']

            if self.register_store['use_parallel']:
                break

            self.noises_pred += [noise_pred]
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        if self.register_store['use_parallel']:
            for i, t in enumerate(self.register_store['timesteps']):
                curr_noise = noise_pred[i * self.opt.batch_size: (i + 1) * self.opt.batch_size]
                self.noises_pred += [curr_noise]
                latents = self.scheduler.step(curr_noise, t, latents)['prev_sample']

        return latents

    def prompt_to_img_student(self, student_unet, prompts, negative_prompts='', height=512, width=512, num_inference_steps=4, latents=None):

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
        latents = self.produce_latents_student(student_unet, text_embeddings, height=height, width=width, num_inference_steps=num_inference_steps, latents=latents)  # [1, 4, 64, 64]

        # # Img latents -> imgs
        # imgs = self.decode_latents(latents)  # [1, 3, 512, 512]
        #
        # # Img to Numpy
        # from PIL import Image
        # imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        # imgs = (imgs * 255).astype('uint8')
        # imgs = self.image_grid(imgs, grid_size=(imgs.shape[0], 1))
        # imgs = Image.fromarray(imgs)
        # imgs.save(f'student.jpg')

        return latents, text_embeds

    def image_grid(self, img, grid_size):
        gw, gh = grid_size
        _N, H, W, C = img.shape
        img = img.reshape(gh, gw, H, W, C)
        img = img.transpose(0, 2, 1, 3, 4)
        img = img.reshape(gh * H, gw * W, C)
        return img

    def kl_loss(self, noise=None):
        """
        Computes the KL divergence between the noise and the target distribution.

        Args:
        noise: The generated noise.
        target_dist: The target distribution, mean is 0 and standard deviation is 1.

        Returns:
        The KL divergence.
        """

        if not noise:
            noise = self.noises_pred

        total_kl_div = 0
        if self.opt.lambda_kl > 0.0:
            for n in noise:
                mu = torch.mean(n)
                logvar = torch.std(n).pow(2).log()
                kl_div = torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * (-0.5 * self.opt.lambda_kl)
                total_kl_div += kl_div

        total_dis_loss = 0
        if self.opt.lambda_dis > 0.0:
            for i, n in enumerate(noise):
                for j, m in enumerate(noise):
                    if i < j:
                        dis_loss = - 0.5 * F.mse_loss(n, m, reduction="mean") * self.opt.lambda_dis
                        total_dis_loss += dis_loss
        return total_kl_div

if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='A pineapple', type=str,)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    opt.t_range = [0.02, 0.98]
    sd = StableDiffusion(device, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
