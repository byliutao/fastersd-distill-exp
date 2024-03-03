import torch
import json
import clip
from PIL import Image
import os
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from pipeline_rf import RectifiedFlowPipeline
import random
import hpsv2

random.seed(2024)

def get_clip_score(image, text):
    # Load the pre-trained CLIP model and the image

    # Preprocess the image and tokenize the text
    image_input = clip_preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])
    
    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()
    
    return clip_score

## load clip model
clip_model, clip_preprocess = clip.load('ViT-L/14@336px')
clip_model = clip_model.to("cuda:0")

## load prompt
coco_f = open('/data/dataset/coco2014-val/annotations/captions_val2014.json')
coco_annotations = json.load(coco_f)
captions = []
for annotation in coco_annotations['annotations']:
    caption = annotation['caption']
    captions.append(caption)
coco_f.close()
captions_30k = random.choices(captions, k=30000)


