import requests
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("/data/model/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("/data/model/blip-image-captioning-large").to("cuda")

print("load model sucess")


def load_images_from_directory(directory):
    image_list = []
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more file extensions if needed
            image_path = os.path.join(directory, filename)
            
            # Open and append the image to the list
            try:
                img = Image.open(image_path)
                image_list.append(img)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

    return image_list

def image_to_text(image, condtion_text=None):
    if(condtion_text is not None):
        # conditional image captioning
        inputs = processor(image, condtion_text, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    else:
        # unconditional image captioning
        inputs = processor(image, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True) 
    

def save_image_with_text(images, save_dir):
    for image in images:
        text = image_to_text(image, "a photography of")
        save_path = os.path.join(save_dir,text+".jpg")
        image.save(save_path)
    

if __name__ == "__main__":
    images = load_images_from_directory("/home/liutao/workspace/distill/swift_photo")
    save_image_with_text(images, "/home/liutao/workspace/distill/swift_photo_with_text")

