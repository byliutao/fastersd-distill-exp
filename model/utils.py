import os
# import glob
import tqdm
import random
import numpy as np
import copy

import time
import gc
# import cv2
import wandb
from PIL import Image

import torch
# import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torch.distributed as dist
# import torchvision
from torch_ema import ExponentialMovingAverage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # pipelines
# from diffusers.utils import deprecate
from diffusers.utils.import_utils import is_xformers_available
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

# torch.backends.cuda.matmul.allow_tf32 = True

import sys
sys.path.append("..")

from utils.embed_utils import get_hash_value


class Trainer(object):
    def __init__(self, 
		         argv, # command line args
                 name, # name of this experiment
                 opt, # extra conf
                 student_unet, # student
                 teacher, # teacher network
                 ema_decay=None, # if use EMA, set the decay
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 fp16=False, # amp optimize level
                 workspace='workspace', # workspace to save logs & ckpts
                 scheduler_update_step=500, # call scheduler.step() after train step
                 freeze_portion_student=False,
                 epoch=0,
                 local_step=0,
                 global_step=0
                 ):
        
        self.argv = argv
        self.name = name
        self.opt = opt
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_step = scheduler_update_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

        # Freeze portion of student
        if freeze_portion_student:
            print(f'[INFO] Freeze portion of student')
            for name, param in student_unet.named_parameters():
                # Freeze the resnet modules in the encoder of self.student_unet using set literals {}.
                if {'down_blocks', 'resnets'}.issubset(name.split('.')) and param.shape[0] >= 1280:
                    param.requires_grad_(False)
                # if {'conv_in', 'time_proj', 'time_embedding', 'conv_norm_out', 'conv_act', 'conv_out'}.intersection(name.split('.')) != set():
                #     param.requires_grad_(False)

        if self.world_size > 1:
            student_unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_unet)
            student_unet = torch.nn.parallel.DistributedDataParallel(student_unet, device_ids=[local_rank])
            teacher.lora_layers = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher.lora_layers)
            teacher.lora_layers = torch.nn.parallel.DistributedDataParallel(teacher.lora_layers, device_ids=[local_rank])
        self.student_unet = student_unet

        # T2I Teacher and LoRA Teacher
        self.teacher = teacher

        # lr=1e-3 for prolificdreamer, le=1e-6 for SwiftBrush
        self.optimizer = optim.Adam(self.student_unet.parameters(), lr=opt.student_lr)
        if opt.lr_scheduler:
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 0.99)

        if not opt.lora:
            self.lora_optimizer = optim.Adam(self.teacher.lora_unet.parameters(), lr=self.opt.unet_lr, betas=(0.9, 0.99), weight_decay=1e-2, eps=1e-8)
        else:
            self.lora_optimizer = optim.AdamW(self.teacher.lora_layers.parameters(), lr=self.opt.unet_lr, betas=(0.9, 0.99), weight_decay=1e-2, eps=1e-8)

        if opt.lr_scheduler:
            self.lora_scheduler = optim.lr_scheduler.LambdaLR(self.lora_optimizer, lr_lambda=lambda step: 0.99 ** step)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.student_unet.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = epoch
        self.global_step = global_step
        self.local_step = local_step

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(f'[INFO] Opt: {opt}')
        self.log(f'[INFO] Cmdline: {self.argv}')
        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(
            f'[INFO] #parameters (student unet): {sum([p.numel() for p in self.student_unet.parameters() if p.requires_grad])}')
        self.log(
            f'[INFO] #parameters (lora layers): {sum([p.numel() for p in self.teacher.lora_layers.parameters() if p.requires_grad])}')

        # loss curve
        if opt.save_loss_fig:
            self.colors = [(220 / 255, 57 / 255, 18 / 255), (1, 153 / 255, 0), (16 / 255, 150 / 255, 24 / 255),
                      (153 / 255, 0, 153 / 255), (0, 153 / 255, 198 / 255), (221 / 255, 68 / 255, 119 / 255),
                      (102 / 255, 170 / 255, 0), (51 / 255, 102 / 255, 204 / 255)]
            fig, axs = plt.subplots(1, 3, figsize=(20, 5))
            axs[0].plot([], label='vsd loss', color=self.colors[local_rank], alpha=0.5)
            axs[1].plot([], label='lora loss', color=self.colors[local_rank], alpha=0.5)
            axs[2].plot([], label='kl loss', color=self.colors[local_rank], alpha=0.5)
            self.axs = axs

    def log(self, *args):
        if self.local_rank == 0:
            if self.log_ptr:
                print(*args)
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file


    def train_step(self, prompt):
        latents_student, text_embedding = self.teacher.prompt_to_img_student(self.student_unet, prompt, num_inference_steps=self.opt.num_inference_steps)

        loss, latents = self.teacher.train_step(text_embedding, latents_student, self.opt.cfg, self.opt.lc_augment)

        return text_embedding, loss, latents


    def train(self, max_epochs):

        start_t = time.time()

        if self.opt.save_loss_fig:
            self.axs[0].set_xlim(0, int(self.opt.max_epoch * len(self.train_loader)))
            self.axs[1].set_xlim(0, int(self.opt.max_epoch * len(self.train_loader)))
            self.axs[2].set_xlim(0, int(self.opt.max_epoch * len(self.train_loader)))

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch()

        if self.local_rank == 0:
            self.save_checkpoint(name='best')

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t) / 60:.4f} minutes.")
    
    def train_one_epoch(self):
        self.log(f"==> Start Training {self.workspace}, Epoch {self.epoch}, "
                 f"vsd_loss lr={self.optimizer.param_groups[0]['lr']:.6f}, lora_loss lr={self.lora_optimizer.param_groups[0]['lr']:.6f}")

        self.student_unet.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(self.train_loader),
                             bar_format='{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        self.drop_condition = np.random.choice([0, 1], len(self.train_loader)*self.opt.batch_size, p=[0.1, 0.9])

        for i, prompt in enumerate(self.train_loader):
            pth_path = [f'{self.opt.embed_path}/{get_hash_value(p)}.pth' for p in prompt]
            self.teacher.pth_path = pth_path
            if self.opt.use_embeddings and False in [os.path.exists(pth) for pth in pth_path]:
                continue

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()  # for student_unet

            with torch.cuda.amp.autocast(enabled=self.fp16):
                text_embedding, loss, latents = self.train_step(prompt)
                loss_item = loss.detach().item()
                if self.opt.use_wandb:
                    wandb.log({f'{self.opt.loss_type} loss': loss_item})
                if self.opt.save_loss_fig:
                    self.axs[0].plot(self.global_step, loss_item, '.-', color=self.colors[self.local_rank], alpha=0.5)

                if self.opt.lambda_kl > 0.0 or self.opt.lambda_dis > 0.0:
                    loss_kl = self.teacher.kl_loss()
                    if self.opt.use_wandb:
                        wandb.log({f'kl loss': loss_kl.detach().item()})
                else:
                    loss_kl = torch.tensor(0)
                kl_loss_item = loss_kl.detach().item()
                if self.opt.save_loss_fig:
                    self.axs[1].plot(self.global_step, kl_loss_item, '.-', color=self.colors[self.local_rank], alpha=0.5)

            loss = loss + loss_kl

            # loss.backward()
            # self.optimizer.step()
            self.scaler.scale(loss).backward()  # 2336MB
            self.scaler.step(self.optimizer)  # 12004MB
            self.scaler.update()

            if self.opt.lr_scheduler and self.global_step % self.scheduler_update_step == 0:
                self.scheduler.step()

            torch.cuda.empty_cache()

            if not self.opt.sds:
                self.lora_optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    timesteps = torch.randint(0, 1000, (self.opt.batch_size,), device=self.device).long() # temperarily hard-coded for simplicity
                    latents_clean = latents.detach()
                    noise = torch.randn(latents_clean.shape, device=self.device, dtype=self.opt.dtype)
                    latents_noisy = self.teacher.scheduler.add_noise(latents_clean, noise, timesteps)
                    uncond_embedding, text_embedding = text_embedding.chunk(2)
                    index = self.drop_condition[(self.local_step-1)*self.opt.batch_size: self.local_step*self.opt.batch_size]
                    text_embedding = torch.stack(
                        [text_embedding[i] if idx == 1 else uncond_embedding[i] for i, idx in enumerate(index)])

                    lora_output = self.teacher.lora_unet(latents_noisy, timesteps, text_embedding).sample  # 6364MB
                    lora_loss = 0.5 * F.mse_loss(lora_output, noise, reduction="mean")
                    lora_loss_item = lora_loss.detach().item()
                    if self.opt.use_wandb:
                        wandb.log({'lora loss': lora_loss_item})
                    if self.opt.save_loss_fig:
                        self.axs[2].plot(self.global_step, lora_loss_item, '.-', color=self.colors[self.local_rank], alpha=0.5)
                # lora_loss.backward()
                # self.unet_optimizer.step()
                self.scaler.scale(lora_loss).backward()  # 338MB
                self.scaler.step(self.lora_optimizer)
                self.scaler.update()
                if self.opt.lr_scheduler and self.global_step % self.scheduler_update_step == 0:
                    self.lora_scheduler.step()

                torch.cuda.empty_cache()

            if self.opt.save_loss_fig:
                plt.savefig(f'{self.workspace}/loss_curve_{self.local_rank}.png', bbox_inches='tight')

            if self.global_step % 2000 == 0:
                if self.opt.init_lora:
                    # initialize lora_layers
                    if self.opt.world_size > 1:
                        self.teacher.lora_layers.module.load_state_dict(self.teacher.init_lora_layers.state_dict(), strict=True)
                    else:
                        self.teacher.lora_layers.load_state_dict(self.teacher.init_lora_layers.state_dict(), strict=True)
                    # reset the optimizer
                    self.lora_optimizer.zero_grad()
                    self.lora_optimizer.step()
                # save LoRA Teacher and Student model
                self.save_checkpoint()

            if self.local_rank == 0:
                pbar.set_description(f"Epoch{self.epoch} "
                                     f"[{self.opt.loss_type} loss={loss_item:.4f}, kl loss={kl_loss_item:.4f}, "
                                     f"lr={self.optimizer.param_groups[0]['lr']:.8f}], "
                                     f"[lora loss={lora_loss_item:.4f}, lr={self.lora_optimizer.param_groups[0]['lr']:.6f}]")
                pbar.update(1)

                if self.global_step % 500 == 0 or self.global_step in [1, 10, 100]:
                    # save samples
                    imgs = self.save_image(latents, 'Student_samples')
                    if self.opt.use_wandb:
                        wandb.log({f'{self.global_step}.jpg': wandb.Image(imgs)})

                    with torch.no_grad():
                        latents_lora, latents_t2i = [], []
                        for bs, test_prompt in enumerate(self.test_loader):
                            latents_lora += [self.teacher.prompt_to_img(test_prompt, teacher='LoRATeacher')]
                            latents_t2i += [self.teacher.prompt_to_img(test_prompt, teacher='T2ITeacher')]
                            if bs == 2:
                                break
                        _ = self.save_image(torch.cat(latents_lora), f'LoRATeacher_samples')
                        _ = self.save_image(torch.cat(latents_t2i), f'T2ITeacher_samples')

            if self.ema is not None and self.global_step * self.opt.batch_size * self.opt.world_size % 64 == 0:
                self.ema.update()

        if self.local_rank == 0:
            pbar.close()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def save_image(self, latents, path):
        latents = copy.deepcopy(latents.detach())
        imgs = self.teacher.decode_latents(latents).cpu()  # [1, 3, 512, 512]
        imgs = imgs.permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')
        out_path = os.path.join(self.workspace, f'{path}_{self.opt.loss_type}')
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
        imgs = self.teacher.image_grid(imgs, grid_size=(self.opt.batch_size, int(imgs.shape[0]/self.opt.batch_size)))
        imgs = Image.fromarray(imgs)
        imgs.save(f'{out_path}/{self.global_step}.jpg')
        return imgs

    def save_checkpoint(self, name=None):
        state = {'epoch': self.epoch, 'local_step': self.local_step, 'global_step': self.global_step,
                 'lr_scheduler': self.opt.lr_scheduler, f'{self.opt.loss_type}_lr': self.opt.student_lr,
                 'lora_lr': self.opt.unet_lr}
        if self.opt.world_size > 1:
            state['student_unet'] = self.student_unet.module.state_dict()
            state['lora_layers'] = self.teacher.lora_layers.module.state_dict()
        else:
            state['student_unet'] = self.student_unet.state_dict()
            state['lora_layers'] = self.teacher.lora_layers.state_dict()
        if self.ema is not None:
            state['student_unet_ema'] = self.ema.state_dict()
        file_path = f"{self.opt.loss_type}_global_step{self.global_step}.pth"
        if name:
            base, ext = os.path.splitext(file_path)
            file_path = f"{base}_{name}{ext}"
        torch.save(state, os.path.join(self.ckpt_path, file_path))
