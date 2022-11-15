import torch
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import torchvision
from model.IAT_main import IAT


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

model = IAT(type='exp').cuda()
model.load_state_dict(torch.load("best_Epoch_exposure.pth"))
model.eval()

image_paths = [os.path.join(args.input_dir, img_path) for img_path in os.listdir(args.input_dir) if img_path[0] != "."]
os.makedirs(args.output_dir, exist_ok=True)

with torch.no_grad():
    for image_path in tqdm(iterable=image_paths, desc="Running IATNet...", total=len(image_paths)):
        # Read & transform images
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
        image = torch.stack([image])
        image = image.cuda()

        mul, add, enhanced_img = model(image)

        image_out_path = os.path.join(args.output_dir, os.path.basename(image_out_path))
        torchvision.utils.save_image(enhanced_img, image_out_path)
