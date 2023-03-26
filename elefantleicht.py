import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image

#methods = {"gradcam": GradCAM}
model = models.resnet50(pretrained=True)
target_layers = [model.layer4]
img = cv2.imread('C:\\Users\\altipair\\Desktop\\bachelor\\ausgabe\\fake_sample_bild_24.png', 1)[:, :, ::-1]
img = np.float32(img) / 255
input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
targets = None
#cam_algorithm = methods['gradcam']
with GradCAM(model=model,target_layers=target_layers,use_cuda=torch.cuda.is_available()) as cam:
    cam.batch_size = 32
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        aug_smooth=False,
                        eigen_smooth=False)
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

cv2.imwrite('Grad2n_cam.jpg', cam_image)


