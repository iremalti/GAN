from pytorch_grad_cam import GradCAM
import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import cv2


import torch
from torch import nn
from torchvision.utils import save_image
from torchvision import transforms
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM


import os
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import  List
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from PIL import Image
import requests
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


CUDA = True
DATA_PATH = 'C:\\Users\\altipair\\Desktop\\Data\\Bilder'
OUT_PATH = 'C:\\Users\\altipair\\Desktop\\bachelor\\ausgabe'
output_folder = 'C:\\Users\\altipair\\Desktop\\bachelor\\CAM'
LOG_FILE = os.path.join(OUT_PATH, 'log.txt')
BATCH_SIZE = 128
IMAGE_CHANNEL = 3
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 64
D_HIDDEN = 64
EPOCH_NUM = 2
REAL_LABEL = 1.
FAKE_LABEL = 0.
lr = 2e-4
seed = 1

# utils.clear_folder(OUT_PATH)
#print("Logging to {}\n".format(LOG_FILE))
#sys.stdout = utils.StdOut(LOG_FILE)
CUDA = CUDA and torch.cuda.is_available()
#print("Cuda is available = ", torch.cuda.is_available())

#print("Pytorch version: {} ".format(torch.__version__))
#if CUDA:
 #   print("CUDA version: {}\n".format(torch.version.cuda))
if seed is None:
    seed = np.random.randint(1, 10000)
#print("Random Seed: ", seed)
np.random.seed(seed)
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)
cudnn.benchmark =True
device = torch.device("cuda:0" if CUDA else "cpu")


#Generator

class Generator (nn.Module):
    def __init__(self):
        super(Generator, self).__init__()       #selbst implementierter generator  kommt in die klasse nn.Modul
        self.main = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8 , 4, 1, 0, bias = False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(G_HIDDEN *4),
            nn.ReLU(True),
            nn.ConvTranspose2d(G_HIDDEN * 4,G_HIDDEN * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias = False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True),
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, 4, 2, 1, bias = False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Diskriminator

class Diskriminator(nn.Module):
    def __init__(self):
        super(Diskriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.main(x).view(-1, 1).squeeze(1)
        return x


if __name__ == '__main__':

    netG = Generator().to(device)
    netG.apply(weights_init)
 #   print(netG)

    netD = Diskriminator().to(device)
    netD.apply(weights_init)
   # print(netD)

  #  print(netD.main[-2])

    criterion = nn.BCELoss()

    optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (0.5, 0.999))

    dataset = dset.ImageFolder(root = DATA_PATH,
                             transform = transforms.Compose([
                                 transforms.Resize(X_DIM),
                                 transforms.CenterCrop(X_DIM),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ]))


    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

    viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device = device)
    for epoch in range (EPOCH_NUM):
        for i, data in enumerate(dataloader):
            x_real = data[0].to(device)
            real_label = torch.full((x_real.size(0),), REAL_LABEL, device = device)
            fake_label = torch.full((x_real.size(0),), FAKE_LABEL, device = device)

            netD.zero_grad()
            y_real = netD(x_real)
            loss_D_real = criterion(y_real, real_label)
            loss_D_real.backward()

            z_noise = torch.randn(x_real.size(0), Z_DIM, 1, 1, device = device)
            x_fake = netG(z_noise)
            y_fake = netD(x_fake.detach())
            loss_D_fake = criterion(y_fake, fake_label)
            loss_D_fake.backward()
            optimizerD.step()

            netG.zero_grad()
            y_fake_r = netD(x_fake)
            loss_G = criterion(y_fake_r, real_label)
            loss_G. backward()
            optimizerG.step()

            if i % 100 == 0:
                print('Epoche {}[{}/{}] loss_D_real: {:.4f} loss_D_fake: {:.4f} loss_G: {:.4f}'.format(epoch, i, len(dataloader),
                                                                                                loss_D_real.mean().item(),
                                                                                                loss_D_fake.mean().item(),
                                                                                                loss_G.mean().item(), ))

            if i % 100 == 0:
                ...
                vutils.save_image(x_real, os.path.join(OUT_PATH, 'real_sample_bild.png'), normalize = True)
                with torch.no_grad():
                    viz_sample = netG(viz_noise)
                    vutils.save_image(viz_sample, os.path.join(OUT_PATH, 'fake_sample_bild_{}.png'.format(epoch)),normalize=True)
                    torch.save(netG.state_dict(), os.path.join(OUT_PATH, 'netG_{}.pth'.format(epoch)))
                    torch.save(netD.state_dict(), os.path.join(OUT_PATH, 'netD_{}.pth'.format(epoch)))

                # Load the GAN model
                D = netD
                target_layers = []
                target_layers.append(netD.main[-2])
                # Define a Grad-CAM object for the Discriminator
                cam = GradCAM(model=D, target_layers=target_layers)
                img = cv2.imread('C:\\Users\\altipair\\Desktop\\bachelor\\imagenet\\000001.jpg', 1)[:, :, ::-1]
                img = cv2.resize(img, (224, 224))
                img = np.float32(img) / 255

                # Calculate the gradients and activations
                img = torch.Tensor(img).unsqueeze(1)

                mask, logit = cam(img)
                weights = np.mean(mask, axis=(1, 2))
                activations = D(img).detach().numpy()[0]

                # Calculate the weighted activations
                weighted_activations = np.zeros((activations.shape[1], activations.shape[2]))
                for i, weight in enumerate(weights):
                    weighted_activations += weight * activations[i]

                # Normalize the weighted activations
                weighted_activations -= np.min(weighted_activations)
                weighted_activations /= np.max(weighted_activations)

                # Apply Grad-CAM on the image
                cam_img = cv2.resize(img.squeeze().permute(1, 2, 0).numpy(), (64, 64))
                cam_img = cam(gcam=weighted_activations, x=cam_img)

                # Save the Grad-CAM image
               # save_image(torch.from_numpy(cam_img), 'gradcam.png')
                cv2.imwrite(os.path.join(output_folder, 'Grad_cam.jpg'), cam_img)



                #nach bild generierung

                # Berechnung der GradCAM-Map
                cam = grad_cam(img_tensor=img_tensor)

                # Anwendung der GradCAM-Map auf das Eingangsbild
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                result = heatmap * 0.3 + img * 0.5

                # Anzeigen des Ergebnisses
                cv2.imshow('GradCAM Result', result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # das letzte Layer Ihres GAN-Modells als target_layer in der GradCAM-Methode definieren.


