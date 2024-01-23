import hashlib
from tqdm import tqdm
import os
import glob
import torch

import threading

def get_hash_value(original_filename):
    hash_object = hashlib.md5(original_filename.encode())
    hash_value = hash_object.hexdigest()
    return hash_value

def preprocess_embedding(trainer, embed_path='embeddings_store'):
    if not os.path.exists(embed_path):
        os.makedirs(embed_path, exist_ok=True)
    else:
        # pth_files = glob.glob(os.path.join(embed_path, '*.pth'))
        # if len(pth_files) >= len(trainer.train_loader) * trainer.opt.batch_size:
        #     print(f'All the prompts have been saved as .pth files: {len(pth_files)}')
        #     return
        pass

    device = trainer.device
    process_data(trainer.guidance, trainer.test_loader.dataset, embed_path, device)
    process_data(trainer.guidance, trainer.train_loader.dataset, embed_path, device)

def process_data(guidance, data, embed_path, device):
    for prompt in tqdm(data):
        hash_value = get_hash_value(prompt)
        if os.path.exists(f'{embed_path}/{hash_value}.pth'):
            continue
        text_input = guidance.tokenizer(prompt, padding='max_length',
                                                max_length=guidance.tokenizer.model_max_length,
                                                truncation=True, return_tensors='pt')
        with torch.no_grad():
            text_embeddings = guidance.text_encoder(text_input.input_ids.to(device))[0]

        torch.save(text_embeddings.cpu(), f'{embed_path}/{hash_value}.pth')
        loaded_embeddings = torch.load(f'{embed_path}/{hash_value}.pth')

def parallel_preprocess_embedding(trainer, embed_path='embeddings_store', tnum=8):
    if not os.path.exists(embed_path):
        os.makedirs(embed_path, exist_ok=True)
    else:
        # pth_files = glob.glob(os.path.join(embed_path, '*.pth'))
        # if len(pth_files) >= len(trainer.train_loader) * trainer.opt.batch_size:
        #     return
        pass

    device = trainer.device
    data = trainer.train_loader.dataset
    threads = []
    chunk_size = int(len(data) / tnum)

    for i in range(tnum):
        chunk = data[i * chunk_size:(i + 1) * chunk_size]
        thread = threading.Thread(target=process_data, args=(trainer.teacher, chunk, embed_path, device))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()