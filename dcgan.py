import utils
import sys
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch
#import cudnn




CUDA= True
data_path = '~/Data/mnist'  # generierte daten mit jeweiligen ordnern
out_path = 'output'  # generiert output mit jeweiligen ordnern
logfile = os.path.join(out_path, 'log.txt')  # gibt die verknüpfungen und die mitglieder zurück
batch_size = 128  # stapel größe
image_channel = 1   # schwarz weis
# anzahl von color-channels hier 1, weil alle  bilder in mnist single channel sind
z_dim = 100
g_hidden = 64
x_dim = 64
d_hidden = 64
epoch_num = 25  # anzahl Epochen in denen es trainiert
Real_label = 1.  # wahr
Fake_label = 0.  # falsch
lr = 2e-4  # learning rate
seed = 1

#utils.clear_folder(out_path)
# ordner leeren
print("Logging to {}\n".format(logfile))
# sys.stdout = utils.StdOut(logfile)
# stdout als ausgabe des systems
print("Pytorch version {}\n".format(torch.__version__))  # versions angabe
if CUDA:
    print("CUDA version: {}\n".format(torch.__version__))
if seed is None:
    seed = np.random.randint(1, 10000)  # generiert eine zufallsvariable
print("Random Seed: ", seed)  # gibt diese aus
np.random.seed(seed)
torch.manual_seed(seed)  # gibt ein torch.generator objekt zurück
if CUDA:
    torch.cuda.manual_seed(seed)
#cudnn.benchmark =True
device = torch.device("cpu")  # um festzulegen, wo Tensoren und Modelle erstellt werden


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # ruft immer sich selbst auf
        self.main = nn.Sequential(# 1.Schicht
            # verkettet die Ausgaben mit den Eingaben für jedes nachfolgende Modul [nacheinander ausführung]
            nn.ConvTranspose2d(z_dim, g_hidden * 8, 4, 1, 0, bias=False),  # g_hidden * 8 = 64 * 8 = 512
            # Modul kann als der Gradient von Conv2d in Bezug auf seine Eingabe angesehen werden
            # z_dim Anzahl der Kanäle im Eingabebild
            # g_hidden *8  Anzahl der Kanäle, die durch die Faltung erzeugt werden
            # 4 ist die Größe des sich drehenden Kernels bleibt immer gleich
            # 1 ist faltungsschritt standardgemäß 1
            # 0  Zusätzliche Größe, die auf einer Seite jeder Dimension in der Ausgabeform hinzugefügt wird st:0
            # bias =False  bedeutet ohne eine erlenbare voreingenommenheit
            # faltung der Bildebenen
            nn.BatchNorm2d(g_hidden * 8),
            # Eingaben der Schichten werden durch Neuzentrierung und Neuskalierung normalisiert  hier: 512
            #es wird schneller und stabiler
            nn.ReLU(True),  # 2 schicht
            # Eingabe wird direkt geändert, ohne dass eine zusätzliche Ausgabe zugewiesen wird
            # Wendet die gleichgerichtete lineare Einheitsfunktion elementweise an
            # aktivierungsfunktion
            nn.ConvTranspose2d(g_hidden * 8, g_hidden * 4, 4, 2, 1, bias=False),
            #2 ist faltungsschritt
            #wieso die eins ?
            nn.BatchNorm2d(g_hidden * 4),
            nn.ReLU(True),  # 3 schicht
            nn.ConvTranspose2d(g_hidden * 4, g_hidden * 2, 4, 2, 1, bias=False),    # g_hidden * 4 = 256
            nn.BatchNorm2d(g_hidden * 2),
            nn.ReLU(True),  # 4.schicht
            nn.ConvTranspose2d(g_hidden * 2, g_hidden, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden),
            nn.ReLU(True),  # Ausgabe schicht hat keinen batchnorm dafür die weights_init fkt
            nn.ConvTranspose2d(g_hidden, image_channel, 4, 2, 1, bias=False),
            nn.Tanh()  # letzte aktivierungsfunktion sollte man lieber nicht benutzen[buch]
        )
    #führt den generator aus
    def forward(self, y):
        return self.main(y)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(# 1 schicht
            nn.Conv2d(image_channel, d_hidden, 4, 2, 1, bias=False),        # analog zu generator
            nn.LeakyReLU(0.2, inplace=True),    #0.2 Steuert den Winkel der negativen Steigung  an ort und stelle
            # 2 schicht
            nn.Conv2d(d_hidden, d_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3 schicht
            nn.Conv2d(d_hidden * 2, d_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 schicht
            nn.Conv2d(d_hidden * 4, d_hidden * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # ausgabe schicht
            nn.Conv2d(d_hidden * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, y):
        return self.main(y).view(-1, 1).squeeze(1)
    # view Wert für diese Dimension wird abgeleitet, so dass die Anzahl der Elemente in der Ansicht der ursprünglichen
    # Anzahl der Elemente entspricht. das bedeutet Dim( WERT x 1)

    # squeeze Entfernt Dimensionen der Größe 1 aus der Form eines Tensors

def weightsinit(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.2) # abweichung von 0.02
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.2)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    netG = Generator().to(device)           # generator inizialisieren
    netG.apply(weightsinit)                 # anwendung der gewichtung
    print(netG)                             # printen

    netD = Discriminator().to(device)       # generator inizialisieren
    netD.apply(weightsinit)                 # anwendung der gewichtung
    print(netD)                             # printen

    criterion = nn.BCELoss()
    # fehler zu messen
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))   # optimiert mit dem adam algo.
    # betas sind  Koeffizienten, die für die Berechnung der laufenden Durchschnitte des Gradienten und seines Quadrats
    # verwendet werden

    dataset = dset.MNIST(root=data_path, download=True,
                         transform=transforms.Compose([
                             transforms.Resize(x_dim),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))   # 1channel
                         ]))
    # transforms.ToTensor()konvertiert die Daten in einen PyTorch-Tensor. koeffizient 0-1 problem schwarz und weis mit
    # hintergrund schwarz tendiert zu 0
    # transforms. Normalize()wandelt den Bereich der Tensor koeffizienten um. koeffizient -1-1 durch  -0.5 und /0.5
    # wird von 0 bis 1 zu -1 bis 1
    # download=True Datensatz bei der ersten Ausführung des obigen Codes heruntergeladen und im aktuellen Verzeichnis
    # gespeichert wird, das durch das Argument root angegeben wird. train=True --> Wenn True, wird der Datensatz aus
    # train-images-idx3-ubyte erstellt, ansonsten aus t10k-images-idx3-ubyte.

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # Daten aus train_set mischt und Stapel von 32 Proben zurückgibt, die Sie zum Trainieren der neuronalen Netze
    # verwenden werden

    viz_noise = torch.randn(batch_size, z_dim, 1, 1, device=device)     # zufälliges rauschen
    # batch size  bestimmt den shape des output tensors
    # z dim  ist output tensor
    # 1 da es none entsprechen soll wird ein global default verwendet
    # 1 ist ein faltungsschritt

    for epoch in range(epoch_num):
        for i, data in enumerate(dataloader):          # i wird hochgezählt dabei ist data[0] die erste komponente in dataloader
            x_real = data[0].to(device)                # zum cpu schicken
            real_label = torch.full((x_real.size(0),), Real_label, device=device)
            # offene größe deshalb leer  wird alles als wahr  festgelegt
            fake_label = torch.full((x_real.size(0),), Fake_label, device=device)
            # offene größe deshalb leer  wird alles als falsch festgelegt

    #dikriminator training
            netD.zero_grad()                            # setzt die gradienten auf 0
            y_real = netD(x_real)                       #vom diskriminator generierten daten
            lossD_real = criterion(y_real, real_label)
            # berechnen der loss funktion durch die generierten daten und den wahrheitswert
            lossD_real.backward()                       # Berechnet den Gradienten der aktuellen Tensor
        # alle daten vom loader sind  als wahr makiert worden und dem diskriminator übergeben

            z_noise = torch.randn(x_real.size(0), z_dim, 1, 1, device=device)
            # hier entspricht die größe die der anzahl der daten
            x_fake = netG(z_noise)
            # gerenrator erstellt fake daten
            y_fake = netD(x_fake.detach())  # daten wieder auf gpu
            lossD_fake = criterion(y_fake, fake_label)
            # berechnen der loss funktion durch die generierten falschen daten und den falschwerten
            lossD_fake.backward()
            # Berechnet den Gradienten der aktuellen Tensor
            optimizerD.step()   # führt ein optimierungsschritt durch
             # alle vom generator generierten daten  wurden als  falsch markiert und dem diskriminator übergeben worden

    #generator training
            netG.zero_grad()
            # setzt die gradienten auf 0
            y_faker = netD(x_fake)  # Ausgabe des dikriminator mit den fake daten
            lossG = criterion(y_faker, real_label)
            # verlust funktion durch die fake daten die mit wahrheitswerten entstehen
            lossG.backward()
            # Berechnet den Gradienten der aktuellen Tensor
            optimizerG.step()
            # führt ein optimierungsschritt durch

            if i % 100 == 0:    # i geht in 100er schritten
                print(
                    'Epoche {}[{}/{}] lossDreal: {:.4f} lossDfake: {:.4f} lossG: {:.4f}'.format(epoch, i, len(dataloader),
                                                                                              lossD_real.mean().item(),
                                                                                              lossD_fake.mean().item(),
                                                                                              lossG.mean().item(), ))
                #datensatz größe ist 469

            if i % 100 == 0:
                vutils.save_image(x_real, os.path.join(out_path, 'real_sample.png'), normalize=True)
                #gibt das bild aus dem loader zurück
                with torch.no_grad():
                    viz_sample = netG(viz_noise)
                    vutils.save_image(viz_sample, os.path.join(out_path, 'fake_sample_{}.png'.format(epoch)),normalize=True)
                    #erstellt ein bild der momentanen gelernten phase
                    torch.save(netG.state_dict(), os.path.join(out_path, 'netG_{}.pth'.format(epoch)))
                    torch.save(netD.state_dict(), os.path.join(out_path, 'netD_{}.pth'.format(epoch)))


# disrkriminator:  fake wird als falsch makiert real als wahr beide daten versionen werden jeweils dem sikriminator
#                   übergeben um fest zustellen was real was fake ist und daraussollte der entscheiden ob die daten vom
#                   generator übergeben werden real oder fake sind


#generator: übergibt die generierten daten an den diskriminator und vergleicht diese mit den wahrheitswerten und
#           erstellt dazu eine loss funktion