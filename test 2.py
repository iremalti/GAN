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



# Load the GAN model
G = torch.load('C:\\Users\\altipair\\Desktop\\bachelor\\output\\netD_{}.pth'.format(epoch))

# Define a Grad-CAM object for the Discriminator
cam = GradCAM(model=G.D, target_layer=nn.Sequential(*list(G.D.children())[-2:]))

# Calculate the gradients and activations
mask, logit = cam(img)
weights = np.mean(mask, axis=(1, 2))
activations = G.D(img).detach().numpy()[0]

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
save_image(torch.from_numpy(cam_img), 'gradcam.png')




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


