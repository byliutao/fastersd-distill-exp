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

@click.command()
@click.pass_context
@click.option('--network', 'network_pth', help='Network pickle filename', required=True, type=str,
              default='checkpoints/vsd_global_step11000.pth')
@click.option('--sd_base_path', type=str, default='/data/', help="'/data/' for 3090, '/home/share/' for a40")  # choices=['/data/', '/home/share/', ],
@click.option('--prompts', type=ast.literal_eval,
              default="['A DSLR photo of a dog reading a book', 'A DSLR photo of a cat reading a book', 'A DSLR photo of a owl reading a book', 'A DSLR photo of a panda reading a book']")
@click.option('--save_path', help='folder where to save images', type=str, default='results/others')
@click.option('--seed', help='seed', type=ast.literal_eval, default='2023')

def generate_images(
        ctx: click.Context,
        network_pth: str,
        sd_base_path: str,
        prompts: list,
        save_path: str,
        seed,
):
    vae, tokenizer, text_encoder, unet, alphas = load_model(sd_base_path, network_pth)

    imgs_num = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    start_time = time.time()
    if seed is not list:
        seed = list(range(1, 201))
    for prompt in prompts:
        if not os.path.exists(f'{save_path}/{prompt}'):
            os.makedirs(f'{save_path}/{prompt}', exist_ok=True)
        for s in seed:
            imgs_num += 1
            imgs = prompt_to_img_student((vae, tokenizer, text_encoder, unet), prompt, seed=s, alphas=alphas)
            save_image(imgs, f'{save_path}/{prompt}/seed{s}')

    # # interpolating the noise input
    # for prompt in prompts:
    #     _save_path = f'{save_path}/latents_interpolating/{prompt}'
    #     if not os.path.exists(_save_path):
    #         os.makedirs(_save_path, exist_ok=True)
    #     generator_2023 = torch.Generator().manual_seed(2023)
    #     latents_2023 = torch.randn((1, 4, 64, 64), generator=generator_2023).to(unet.device)
    #     generator_2024 = torch.Generator().manual_seed(2024)
    #     latents_2024 = torch.randn((1, 4, 64, 64), generator=generator_2024).to(unet.device)
    #     for i in np.arange(0.0, 1.0, 0.1):
    #         imgs_num += 1
    #         latents = latents_2023 * (1 - i) + latents_2024 * i
    #         imgs = prompt_to_img_student((vae, tokenizer, text_encoder, unet), prompt, latents=latents, alphas=alphas)
    #         save_image(imgs, f'{_save_path}/output_image_{i:.1f}')

    end_time = time.time()

    print(f'time: {(end_time - start_time):.3f} seconds, average time: {(end_time - start_time)/imgs_num  :.3f}')

def image_grid(img, grid_size):
    gw, gh = grid_size
    _N, H, W, C = img.shape
    img = img.reshape(gh, gw, H, W, C)
    img = img.transpose(0, 2, 1, 3, 4)
    img = img.reshape(gh * H, gw * W, C)
    return img

def save_image(imgs, path):
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    imgs = (imgs * 255).round().astype('uint8')
    imgs = image_grid(imgs, grid_size=(-1, 1))
    imgs = Image.fromarray(imgs)
    imgs.save(f'{path}.jpg')
    return imgs

@torch.no_grad()
def prompt_to_img_student(student, prompt, height=512, width=512, latents=None, seed=2023, alphas=None):
    vae, tokenizer, text_encoder, unet = student

    generator = torch.Generator().manual_seed(seed)

    if isinstance(prompt, str):
        prompt = [prompt]

    # Prompts -> text embeds
    text_input = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_embeddings = text_encoder(text_input.input_ids.to(unet.device))[0]

    # Text embeds -> img latents
    if latents is None:
        latents = torch.randn((text_embeddings.shape[0], unet.config.in_channels, height // 8, width // 8), generator=generator).to(unet.device)

    T = torch.tensor(int(999))

    # predict the noise residual
    noise_pred = unet(latents, T, encoder_hidden_states=text_embeddings)['sample']

    # re-parameterize following SwiftBrush
    alpha_T = alphas[T] ** 0.5
    sigma_T = (1 - alphas[T]) ** 0.5
    latents = (latents - sigma_T * noise_pred) / alpha_T

    # Img latents -> imgs
    latents = 1 / 0.18215 * latents
    imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)

    return imgs

def load_model(sd_base_path, network_pth):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_key = f"{sd_base_path}stabilityai/stable-diffusion-2-1-base"
    vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet")
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    student_data = torch.load(network_pth, map_location=device)
    if 'student_unet' in student_data.keys():
        print(f'[INFO] loading student unet checkpoint')
        unet.load_state_dict(student_data['student_unet'], strict=True)
        for key, param in student_data['student_unet'].items():
            student_data['student_unet'][key] = param.detach().cpu()
        for key, param in student_data['lora_layers'].items():
            student_data['lora_layers'][key] = param.detach().cpu()
        torch.cuda.empty_cache()

    scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    alphas = scheduler.alphas_cumprod.to(device)

    return vae, tokenizer, text_encoder, unet, alphas


if __name__ == "__main__":
    generate_images()
