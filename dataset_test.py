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
import torchvision.datasets as dset
import torchvision.transforms as transforms


DATA_PATH = 'C:\\Users\\altipair\\Desktop\\Data\\Bilder'
output_folder = 'C:\\Users\\altipair\\Desktop\\bachelor\\testordner'
X_DIM = 64
BATCH_SIZE = 128


if __name__ == '__main__':
    dataset = dset.ImageFolder(root=DATA_PATH,
                               transform=transforms.Compose([
                                   transforms.Resize(X_DIM),
                                   transforms.CenterCrop(X_DIM),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = models.resnet50(pretrained=True)
    target_layers = [model.layer4]

    for i, data in enumerate(dataloader):
        img_path, _ = data[i] # Pfad zum Bild und Klassenlabel ignorieren
        img_path_str = str(img_path)
        img = cv2.imread(img_path_str, 1)[:, :, ::-1] # BGR -> RGB
        img = cv2.resize(img, (224, 224))
        img = np.float32(img) / 255
        input_tensor = preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # Hier weitere Verarbeitung mit dem Input Tensor durchführen


    #img = cv2.imread('C:\\Users\\altipair\\Desktop\\bachelor\\imagenet\\000001.jpg',1)[:, :, ::-1]
    #img = np.float32(img) / 255
    #input_tensor = preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        targets= None
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



