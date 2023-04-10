import cv2
import numpy as np
import torch
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
import os
from PIL import Image
import requests

output_folder = 'C:\\Users\\altipair\\Desktop\\bachelor\\CAM'

model = models.resnet50(pretrained=True)
target_layers = [model.layer4]

#image_url = "https://th.bing.com/th/id/R.94b33a074b9ceeb27b1c7fba0f66db74?rik=wN27mvigyFlXGg&riu=http%3a%2f%2fimages5.fanpop.com%2fimage%2fphotos%2f31400000%2fBear-Wallpaper-bears-31446777-1600-1200.jpg&ehk=oD0JPpRVTZZ6yizZtGQtnsBGK2pAap2xv3sU3A4bIMc%3d&risl=&pid=ImgRaw&r=0"
#img = np.array(Image.open(requests.get(image_url, stream=True).raw))
#img = cv2.resize(img, (224, 224))

img = cv2.imread('C:\\Users\\altipair\\Desktop\\bachelor\\imagenet\\000001.jpg',1)[:, :, ::-1]
img = np.float32(img) / 255
input_tensor = preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

targets= None
#targets = [ClassifierOutputTarget(208)]
with GradCAM(model=model,target_layers=target_layers,use_cuda=torch.cuda.is_available()) as cam:
    cam.batch_size = 32
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        aug_smooth=False,
                        eigen_smooth=False)
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

cv2.imwrite(os.path.join(output_folder, 'frau4_cam.jpg'), cam_image)



