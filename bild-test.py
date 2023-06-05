import os
import random

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
from pytorch_grad_cam.utils.model_targets import  BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt



CUDA = True
DATA_PATH = 'C:\\Users\\altipair\\Desktop\\Data\\CelebA'
OUT_PATH = 'C:\\Users\\altipair\\Desktop\\bachelor\\ausgabe_gan'
output_folder = 'C:\\Users\\altipair\\Desktop\\bachelor\\test'
output_folder_fake = 'C:\\Users\\altipair\\Desktop\\bachelor\\fake'
output_folder_real = 'C:\\Users\\altipair\\Desktop\\bachelor\\real'
LOG_FILE = os.path.join(OUT_PATH, 'log.txt')
BATCH_SIZE = 128
IMAGE_CHANNEL = 3
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 64
D_HIDDEN = 64
EPOCH_NUM = 5
REAL_LABEL = 1.
FAKE_LABEL = 0.
lr = 2e-4
seed = 1


CUDA = CUDA and torch.cuda.is_available()
if seed is None:
    seed = random.seed(seed)
   # seed = np.random.randint(1, 10000)
np.random.seed(seed)
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
cudnn.benchmark =True
device = torch.device("cuda:0" if CUDA else "cpu")
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


dataset = dset.ImageFolder(root=DATA_PATH,
                           transform=transforms.Compose([
                               transforms.Resize(X_DIM),
                               transforms.CenterCrop(X_DIM),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


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
    def forward(self, x):
        return self.main(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Diskriminator(nn.Module):
    def __init__(self):
        super(Diskriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x).view(-1, 1).squeeze(1)
        return x


if __name__ == '__main__':

    netG = Generator().to(device)
    netG.apply(weights_init)

    netD = Diskriminator().to(device)
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (0.5, 0.999))

    G_losses = []
    D_losses_real = []
    D_losses_fake = []
    img_list = []
    cam_list = []
    cam_tensor =torch.empty(0, 3, 64, 64)


    noise_list =[]
    bilder = []
    count = 0

    for batch in dataloader:
        images = batch[0].to(device)

        for image in images:
            bilder.append(image)
            count += 1

            if count >= 50:
                break

        if count >= 50:
            break


    for _ in range(50):
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
        noise_list.append(noise)

    viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device = device)
    for epoch in range (EPOCH_NUM):
        set_random_seed(seed)
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
            loss_D_fake.backward(retain_graph=True )
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

                G_losses.append(loss_G.item())
                D_losses_real.append(loss_D_real.item())
                D_losses_fake.append(loss_D_fake.item())

            if i % 100 == 0:

                vutils.save_image(x_real, os.path.join(OUT_PATH, 'real_sample_bild.png'), normalize = True)
                with torch.no_grad():
                    viz_sample = netG(viz_noise)
                    vutils.save_image(viz_sample, os.path.join(OUT_PATH, 'fake_sample_bild_{}.png'.format(epoch)),normalize=True)
                    torch.save(netG.state_dict(), os.path.join(OUT_PATH, 'netG_{}.pth'.format(epoch)))
                    torch.save(netD.state_dict(), os.path.join(OUT_PATH, 'netD_{}.pth'.format(epoch)))


#Cam start
                model = netD
                target_layers = [netD.main[-2]]
                targets = [BinaryClassifierOutputTarget(0)]

                #Realer Datensatz
                bilder_input = torch.stack(bilder)  # Liste in Tensor umwandeln

                for k in range (bilder_input.size(0)):
                    image = bilder_input[k].cpu().numpy()
                    img = image.transpose((1, 2, 0))  # Achsen vertauschen, um die Form (64, 64, 3) zu erhalten
                    org = np.clip(img, 0, 1)  # Werte auf den Bereich von 0-1 begrenzen (falls erforderlich)
                    org = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)  # richtig rum
                    org = (org * 255).astype(np.uint8)  # Skalierung der Werte auf den Bereich von 0-255
                    cv2.imwrite(os.path.join(output_folder_real, 'Bild_real' + str(k) + '.jpg'), org)

                    with GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
                        grayscale_cam = cam(input_tensor=bilder_input,
                                                targets=targets,
                                                aug_smooth=False,
                                                eigen_smooth=False)
                        grayscale_cam = grayscale_cam[0, :]
                        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
                        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(os.path.join(output_folder_real, 'Bild_real{'+str(epoch)+'}_{'+str(i)+'}_{'+str(k)+'}_cam.jpg'), cam_image)
                    cam_image = cam_image.transpose((2, 0, 1))
                    cam_image_tensor = torch.from_numpy(cam_image)
                    image_float = cam_image_tensor.clone().detach().to(torch.float32) / 255.0
                    image_float = image_float.unsqueeze(0)
                    cam_tensor = torch.cat((cam_tensor,image_float ), dim=0)




                #Fake Werte

                fake_input = torch.stack(noise_list)
                for k in range (50):
                    x_fake_cam = netG(noise_list[k])
                    x_fake_cam = x_fake.detach().clone()
                    input_tensor = x_fake_cam



                    img_fake = x_fake_cam.cpu().numpy() [0:1, :, :, :]
                    image_fake = img_fake.reshape(3, 64, 64)
                    image_fake = np.swapaxes(image_fake, 0, 1)
                    image_fake = np.swapaxes(image_fake, 1, 2)

                    with GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:

                        grayscale_cam = cam(input_tensor=input_tensor,
                                            targets=targets,
                                            aug_smooth=False,
                                            eigen_smooth=False)
                        grayscale_cam = grayscale_cam[0, :]
                        cam_image = show_cam_on_image(image_fake, grayscale_cam, use_rgb=True)
                        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    #
                    cv2.imwrite(os.path.join(output_folder_fake, 'Bild_fake{'+str(epoch)+'}_{'+str(i)+'}_{'+str(k)+'}_cam.jpg'), cam_image)



#Plot der Verluste in den jeweiligen Epochen
#    plt.figure(figsize=(10, 5))
#    plt.title("Generator and Discriminator Loss During Training")
#    plt.plot(G_losses, label="G")
#    plt.plot(D_losses_real, label="D_r")
#    plt.plot(D_losses_fake, label="D_f")
#    plt.xlabel("iterations")
#    plt.ylabel("Loss")
#    plt.legend()
#    plt.show()

        plot = vutils.make_grid(cam_tensor, nrow=5, padding=2, normalize=True)
        plotc = plot.cpu()

        grid = vutils.make_grid(bilder_input, nrow=5, padding=2, normalize=True)
        gridc = grid.cpu()

        fig, axes = plt.subplots(2, 1,figsize=(10, 10))

        axes[0].imshow(plotc.permute(1, 2, 0))
        axes[0].axis('off')

        axes[1].imshow(gridc.permute(1, 2, 0))
        axes[1].axis('off')

        plt.subplots_adjust(wspace=0.1)

 #       plt.show()






#    fig, ax = plt.subplots(nrows=16, ncols=len(cam_list))

    # Schleife über die CAMs und zeichnen Sie sie in den Plot
#    for i in range(len(cam_list)):
#        ax[i].imshow(cam_list[i], cmap='jet')  # Verwenden Sie die 'jet'-Farbgebung für CAMs
 #       ax[i].axis('off')  # Optionales Entfernen der Achsenbeschriftungen

    # Zeigen Sie den Plot an
 #   plt.show()







#kopie des gesamten models
              #  model_c = copy.deepcopy(netD) 1 version
              #  model_c = Diskriminator()      Alternative
              #  model_c.load_state_dict(netD.state_dict())
              #  input_tensor = model_c[][0:1, :, :, :]# zu bearbeiten im Model
# x_fake_copy_cpu = x_fake_copy.cpu()
# x_fake_array = x_fake_copy_cpu.numpy()


# Restcode
              #  input = x_fake.detach.cpu()
              #  n_input= input.numpy()
              #  input_tensor = preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # img = cv2.imread('C:\\Users\\altipair\\Desktop\\Data\\CelebA\\img_align_celeba\\000001.jpg',1)[:, :, ::-1]
                #  img = np.float32(img) / 255
                #   img = cv2.resize(img, (64, 64))
                #   realer_tensor = x_real [0:1, :, :, :]
                #  print(realer_tensor.shape)
                #     realer_input = F.interpolate(realer_tensor, size=(128, 128), mode='bilinear', align_corners=False)

                #   img = realer_tensor.cpu().numpy()[0:1, :, :, :]
                #  print(img.shape)
                #    img_tensor = torch.from_numpy(img)
                #   image = F.interpolate(img_tensor, size=(128, 128), mode='bilinear', align_corners=False)
                #  image = image.numpy()
# #               image = np.swapaxes(image,0,1)
#                img = np.swapaxes(image,1,2)
#  image = bilder_input.cpu().numpy()[0]  # Den ersten Eintrag in der Batch-Dimension auswählen
 # cam_image_normalized = cam_image / 255.0
                #    o = (cam_image * 255).astype(np.uint8)  # Skalierung der Werte auf den Bereich von 0-255



# Bildausgabe
                # real_batch = next(iter(dataloader))
                # images = real_batch[0][:10]

                #   data_tensor = torch.cat([batch[0] for batch in dataloader], dim=0)
                #  image= data_tensor.numpy()

                #   for img_path, _ in dataset.imgs:

                #         if i <= 50:
                #            img_path = dataset.imgs[i]  # Pfad zum Bild und Klassenlabel ignorieren
                #           img_path_str = str(img_path)
                #          img = cv2.imread(img_path_str, 1)[:, :, ::-1]  # BGR -> RGB
                #         img = cv2.resize(img, (64, 64))
                #        image = np.float32(img) / 255
                # cam.batch_size = BATCH_SIZE
                # cam.batch_size = 32
    # Ecter Wert , von einem Batch das erste Element
#                realer_input = x_real  # [0:1, :, :, :]
 #              image = realer_input.cpu().numpy()[0:1, :, :, :]
  #              image = image.reshape(3, 64, 64)
   #
    #            image = realer_input.cpu().numpy()[0]  # Den ersten Eintrag in der Batch-Dimension auswählen
     #           img = image.transpose((1, 2, 0))  # Achsen vertauschen, um die Form (64, 64, 3) zu erhalten
      #          org = np.clip(img, 0, 1)  # Werte auf den Bereich von 0-1 begrenzen (falls erforderlich)
       #         org = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)  # richtig rum
        #        org = (org * 255).astype(np.uint8)  # Skalierung der Werte auf den Bereich von 0-255
         #       cv2.imwrite(os.path.join(output_folder, 'Bild_real' + str(i) + '.jpg'), org)
#      image = bilder_input.cpu().numpy()[0:1, :, :, :]
#       image = image.reshape(3, 64, 64)

#to do
            # img = input tensor nur kein tensor
            # auch auf echte Daten

