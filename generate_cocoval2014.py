#!/bin/bash
import hashlib
import numpy as np
import os
import json
import argparse
import random
import torch
import glob
from tqdm import tqdm
import threading
from generate import load_model, prompt_to_img_student, save_image


def get_hash_value(original_filename):
    hash_object = hashlib.md5(original_filename.encode())
    hash_value = hash_object.hexdigest()
    return hash_value

# build category dictionary for coco and lvis
def build_category_dict():
    # if os.path.exists('val2014_captions.npz') is True:  # if .npz file is not exist
    #     npz_file = np.load('val2014_captions.npz', allow_pickle=True)
    #     captions = npz_file['captions'][()]
    #     seeds = npz_file['seeds'][()]
    #     return captions, seeds

    # captions = []
    # # coco annotations file
    # coco_f = open('/data/dataset/coco2014-val/annotations/captions_val2014.json')
    # coco_annotations = json.load(coco_f)
    # for annotation in coco_annotations['annotations']:
    #     caption = annotation['caption']
    #     captions += [caption]
    # coco_f.close()

    # # captions
    # captions = list(set(filter(None, captions)))
    # captions.sort()
    # random.seed(2023)
    # random.shuffle(captions)
    coco_f = open('/data/dataset/coco2014-val/annotations/captions_val2014.json')
    coco_annotations = json.load(coco_f)
    captions = []
    for annotation in coco_annotations['annotations']:
        caption = annotation['caption']
        captions.append(caption)
    coco_f.close()
    random.seed(2024)
    captions = random.choices(captions, k=30000)
    # seeds
    seeds = random.sample(range(0, len(captions)*10), len(captions))
    seeds.sort()
    random.shuffle(seeds)

    np.savez('val2014_captions.npz', captions=captions, seeds=seeds)
    
    return captions, seeds

def process_data(student_model, captions, seeds, save_path):
    vae, tokenizer, text_encoder, unet, scheduler = student_model
    for i, caption in tqdm(enumerate(captions), desc="Processing captions", total=len(captions)):
        hash_value = get_hash_value(caption)
        if os.path.exists(f'{save_path}/{hash_value}.jpg'):
            continue
        seed = int(seeds[i])
        generator = torch.Generator().manual_seed(seed)
        latents = torch.randn((1, 4, 64, 64), generator=generator).to(unet.device)
        imgs = prompt_to_img_student((vae, tokenizer, text_encoder, unet, scheduler), caption, latents=latents)
        save_image(imgs, f'{save_path}/{hash_value}', f'{save_path}/{caption}')

def generate_images(student_model, captions, seeds, save_path):
    process_data(student_model, captions, seeds, save_path)

def parallel_generate_images(student_model, captions, seeds, save_path, pnum=8):
    threads = []
    chunk_size = int(len(captions) / pnum)

    for i in range(pnum):
        chunk = captions[i * chunk_size:(i + 1) * chunk_size]
        chunk_seeds = seeds[i * chunk_size:(i + 1) * chunk_size]
        thread = threading.Thread(target=process_data, args=(student_model, chunk, chunk_seeds, save_path))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def main(args):
    sd_base_path, network_pth, save_path, device = args.sd_base_path, args.network_pth, args.save_path, args.device
    from_case, end_case = args.from_case, args.end_case

    # captions
    captions, seeds = build_category_dict()
    captions = captions[from_case:end_case]
    seeds = seeds[from_case:end_case]

    # student
    vae, tokenizer, text_encoder, unet, scheduler, alphas = load_model(sd_base_path, network_pth, device=device)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # SwiftBrush reproduce with share encoder (Ours)
    generate_images((vae, tokenizer, text_encoder, unet, scheduler), captions, seeds, save_path)
    # parallel_generate_images((vae, tokenizer, text_encoder, unet, scheduler), captions, seeds, save_path, pnum=2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_pth', help='Network pickle filename', type=str, default='/data/20231212/SwiftBrush_reproduce_se_parallel/checkpoints/vsd_global_step8000.pth')
    parser.add_argument('--sd_base_path', type=str, default='/data/', help="'/data/' for 3090, '/home/share/' for a40")
    parser.add_argument('--save_path', help='folder where to save images', type=str, default='/home/liutao/workspace/data/ours_coco30k_test')
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--end_case', help='end generation of case_number', type=int, required=False, default=300_00)  # 30K following SwiftBrush
    parser.add_argument('--device', help='gpu device', type=str, required=False, default="cuda:0")
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    main(args)
