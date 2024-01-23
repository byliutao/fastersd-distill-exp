import os
import random
import json

import re

import torch
from torch.utils.data import DataLoader
import sys
# sys.path.append('../')

class StudentDataset:
    def __init__(self, opt, device, path, world_size=1, num=None):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.world_size = world_size

        self.batch_size = opt.batch_size

        jsonl_file_path = path
        prompt_path = re.sub(r'(train|valid)_anno', r'\1_prompt', path)

        prompts = []
        if not os.path.exists(prompt_path):
            with open(jsonl_file_path, 'r') as jsonl_file:
                for line in jsonl_file:
                    json_object = json.loads(line)
                    key = list(json_object['Task2'].keys())[0]
                    assert key in ['Caption', 'Caption:', 'caption']
                    prompts += [json_object['Task2'][key]]
            prompts = list(set(filter(None, prompts)))
            prompts.sort()
            random.seed(2023)
            random.shuffle(prompts)
            with open(prompt_path, 'w') as jsonl_file:
                json.dump(prompts, jsonl_file)
        else:
            with open(prompt_path, 'r') as jsonl_file:
                prompts = json.load(jsonl_file)

        self.prompts = prompts \
            if num is None else prompts[:num]

    def dataloader(self):
        if self.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.prompts)
            shuffle = False
        else:
            train_sampler = None
            shuffle = True
        loader = DataLoader(self.prompts, batch_size=self.batch_size, drop_last=True, shuffle=shuffle, sampler=train_sampler)
        return loader