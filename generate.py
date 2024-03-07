import click
import time
import torch
import pandas as pd
import os
from PIL import Image
import ast
import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

from utils.se_parallel_utils import register_se_forward  # share encoder

# @click.command()
# @click.pass_context
# @click.option('--network', 'network_pth', help='Network pickle filename', required=True, type=str,
#               default='checkpoints/vsd_global_step8000_ema.pth')
# @click.option('--sd_base_path', type=str, default='/data/', help="'/data/' for 3090, '/home/share/' for a40")  # choices=['/data/', '/home/share/', ],
# @click.option('--prompts', type=ast.literal_eval,
#               default="['A DSLR photo of a dog reading a book', 'A DSLR photo of a cat reading a book', 'A DSLR photo of a owl reading a book', 'A DSLR photo of a panda reading a book', "
#                       "'An oil painting of a train', 'An DSLR photo of a tiger in the city']")
# @click.option('--save_path', help='folder where to save images', type=str, default='results/others')
# @click.option('--seed', help='seed', type=ast.literal_eval, default='100')
# @click.option('--inter', help='interpolating the noise input', type=bool, default=False)

def generate_single_image(
        network,
        prompt: str,
        seed=2024,
        num_inference_steps=4
):
    img = prompt_to_img_student(network, prompt, seed=seed, num_inference_steps=num_inference_steps)
    img = img.permute(0, 2, 3, 1).cpu().numpy()
    img = (img * 255).round().astype('uint8')
    img = image_grid(img, grid_size=(-1, 1))
    img = Image.fromarray(img)
    return img


def generate_images(
        ctx: click.Context,
        network_pth: str,
        sd_base_path: str,
        prompts: list,
        save_path: str,
        seed,
        inter,
        device,
):
    print(network_pth)
    vae, tokenizer, text_encoder, unet, scheduler, alphas = load_model(sd_base_path, network_pth, device=device)

    imgs_num = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    start_time = time.time()

    if not inter:
        # generate images
        if seed is not list:
            seed = list(range(0, seed))
        for prompt in prompts:
            print(f'prompt: {prompt}')
            if not os.path.exists(f'{save_path}/{prompt}'):
                os.makedirs(f'{save_path}/{prompt}', exist_ok=True)
            for s in seed:
                if os.path.exists(f'{save_path}/{prompt}/seed{s}.jpg'):
                    continue
                imgs_num += 1
                imgs = prompt_to_img_student((vae, tokenizer, text_encoder, unet, scheduler), prompt, seed=s)
                save_image(imgs, f'{save_path}/{prompt}/seed{s}')
    else:
        # interpolating the noise input
        for prompt in prompts:
            _save_path = f'{save_path}/latents_interpolating/{prompt}'
            if not os.path.exists(_save_path):
                os.makedirs(_save_path, exist_ok=True)
            generator_1 = torch.Generator().manual_seed(2023)
            latents_1 = torch.randn((1, 4, 64, 64), generator=generator_1).to(unet.device)
            generator_2 = torch.Generator().manual_seed(2024)
            latents_2 = torch.randn((1, 4, 64, 64), generator=generator_2).to(unet.device)
            for i in np.arange(0.0, 1.1, 0.1):
                imgs_num += 1
                # latents = latents_2023 * (1 - i) + latents_2024 * i
                # 归一化张量
                latents_1_normalized = latents_1 / torch.norm(latents_1, p=2, dim=(1, 2, 3), keepdim=True)
                latents_2_normalized = latents_2 / torch.norm(latents_2, p=2, dim=(1, 2, 3), keepdim=True)
                interpolation = slerp(latents_1_normalized, latents_2_normalized, i) * 255
                imgs = prompt_to_img_student((vae, tokenizer, text_encoder, unet, scheduler), prompt, latents=interpolation)
                save_image(imgs, f'{_save_path}/output_image_{i:.1f}')

    end_time = time.time()

    print(f'time: {(end_time - start_time):.3f} seconds, average time: {(end_time - start_time)/imgs_num  :.3f}')

# 定义球面差值函数
def slerp(tensor_A, tensor_B, t):
    # 计算角度θ
    dot_product = (tensor_A * tensor_B).sum((1, 2, 3))
    theta = torch.acos(dot_product).unsqueeze(1).unsqueeze(2).unsqueeze(3)

    # 球面线性插值
    slerp_interpolation = (torch.sin((1 - t) * theta) * tensor_A + torch.sin(t * theta) * tensor_B) / torch.sin(theta)
    return slerp_interpolation

def image_grid(img, grid_size):
    gw, gh = grid_size
    _N, H, W, C = img.shape
    img = img.reshape(gh, gw, H, W, C)
    img = img.transpose(0, 2, 1, 3, 4)
    img = img.reshape(gh * H, gw * W, C)
    return img

def save_image(imgs, path, caption_path=None):
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    imgs = (imgs * 255).round().astype('uint8')
    imgs = image_grid(imgs, grid_size=(-1, 1))
    imgs = Image.fromarray(imgs)

    if caption_path is not None:
        try:
            imgs.save(f'{caption_path}.jpg')
        except:
            print(caption_path)
    else:
        imgs.save(f'{path}.jpg')
    return imgs

@torch.no_grad()
def prompt_to_img_student(student, prompt, height=512, width=512, latents=None, seed=2024, num_inference_steps=4):
    vae, tokenizer, text_encoder, unet, scheduler = student

    generator = torch.Generator().manual_seed(seed)

    if isinstance(prompt, str):
        prompt = [prompt]

    # Prompts -> text embeds
    text_input = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_embeddings = text_encoder(text_input.input_ids.to(unet.device))[0]

    # Text embeds -> img latents
    if latents is None:
        latents = torch.randn((text_embeddings.shape[0], unet.config.in_channels, height // 8, width // 8), generator=generator).to(unet.device)

    scheduler.set_timesteps(num_inference_steps)

    if unet.register_store['use_parallel']:
        unet.register_store['timesteps'] = scheduler.timesteps

    for i, t in enumerate(scheduler.timesteps):
        # predict the noise residual
        unet.register_store['se_step'] = i
        noise_pred = unet(latents, t, encoder_hidden_states=text_embeddings)['sample']

        if unet.register_store['use_parallel']:
            break

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents)['prev_sample']

    if unet.register_store['use_parallel']:
        for i, t in enumerate(unet.register_store['timesteps']):
            curr_noise = noise_pred[i: i + 1]
            latents = scheduler.step(curr_noise, t, latents)['prev_sample']

    # Img latents -> imgs
    latents = 1 / 0.18215 * latents
    imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)

    return imgs

@torch.no_grad()
def prompts_to_noises(student, prompt, height=512, width=512, latents=None, seed=2024, num_inference_steps=4):
    vae, tokenizer, text_encoder, unet, scheduler = student

    generator = torch.Generator().manual_seed(seed)
    noises = []

    if isinstance(prompt, str):
        prompt = [prompt]

    # Prompts -> text embeds
    text_input = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_embeddings = text_encoder(text_input.input_ids.to(unet.device))[0]

    # Text embeds -> img latents
    if latents is None:
        latents = torch.randn((text_embeddings.shape[0], unet.config.in_channels, height // 8, width // 8), generator=generator).to(unet.device)

    scheduler.set_timesteps(num_inference_steps)

    if unet.register_store['use_parallel']:
        unet.register_store['timesteps'] = scheduler.timesteps

    for i, t in enumerate(scheduler.timesteps):
        # predict the noise residual
        unet.register_store['se_step'] = i
        noise_pred = unet(latents, t, encoder_hidden_states=text_embeddings)['sample']

        noises.append({"t": t, "value":noise_pred.cpu()})
        
        if unet.register_store['use_parallel']:
            break

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents)['prev_sample']

    if unet.register_store['use_parallel']:
        for i, t in enumerate(unet.register_store['timesteps']):
            curr_noise = noise_pred[i: i + 1]
            noises.append({"t": t, "value":curr_noise.cpu()})
            latents = scheduler.step(curr_noise, t, latents)['prev_sample']

    return noises



def load_model(sd_base_path, network_pth, device):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model_key = f"{sd_base_path}stabilityai/stable-diffusion-2-1-base"
    vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet")
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    setattr(unet, 'register_store', {'se_step': None, 'skip_feature': None, 'mid_feature': None, 'lora_scale': None,
                                     'use_parallel': False, 'timesteps': [], 'bs': 1})
    unet.register_store['use_parallel'] = True
    register_se_forward(unet, unet.register_store)

    student_data = torch.load(network_pth, map_location=device)
    if 'student_unet' in student_data.keys():
        print(f'[INFO] loading student unet checkpoint')
        unet.load_state_dict(student_data['student_unet'], strict=True)
        for key, param in student_data['student_unet'].items():
            student_data['student_unet'][key] = param.detach().cpu()
        for key, param in student_data['lora_layers'].items():
            student_data['lora_layers'][key] = param.detach().cpu()
        if 'student_unet_ema' in student_data:
            # student_data_ema = {}
            # for key, param in zip(student_data['student_unet'].keys(), student_data['student_unet_ema']['shadow_params']):
            #     assert student_data['student_unet'][key].shape == param.shape
            #     student_data_ema[key] = param
            # unet.load_state_dict(student_data_ema, strict=True)
            for param in student_data['student_unet_ema']['shadow_params']:
                student_data['student_unet_ema']['shadow_params'] = param.detach().cpu()
            # for key, param in student_data_ema.items():
            #     student_data_ema[key] = param.detach().cpu()
        torch.cuda.empty_cache()


    scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    alphas = scheduler.alphas_cumprod.to(device)

    return vae, tokenizer, text_encoder, unet, scheduler, alphas


# if __name__ == "__main__":
#     generate_images()