import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise


from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Erstellen von Argumenten( erster Parameter)
def get_args():
    # Zerlegen in Argumente
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda',
                        action='store_true',
                        default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path',
                        type=str,
                        default='C:\\Users\\altipair\\Desktop\\bachelor\\elefant\\0.jpg',
                        help='Input image path')
    parser.add_argument('--aug_smooth',
                        action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth',
                        action='store_true',
                        help='Reduce noise by taking the first principle componenet of cam_weights*activations')
    parser.add_argument('--method',
                        type=str,
                        default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    # in eine Variable args Zusammensetzen
    args = parser.parse_args()
    args.use_cuda =  torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

# direkte Ausführung und kein Imort
if __name__ == '__main__':
    # Kommandozeilenargumente her holen
    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "hirescam": HiResCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}

    #Model laden resnet50 für convolution neural networks  mit tieferen schichten
    model = models.resnet50(pretrained=True)

    #Schicht wählen hier die vierte
    target_layers = [model.layer4]

    # Lädt das Bild und konvertiert es von BGR in RGB (Farbumdrehung)
    rgb_img = cv2.imread('C:\\Users\\altipair\\Desktop\\bachelor\\elefant\\0.jpg', 1)[:, :, ::-1]

    # Skalierung des Bildes auf ein Bereich 0-1
    rgb_img = np.float32(rgb_img) / 255 # Farbspektrum

    #Erstellung eines Tensors{numpy-array} mean/std skalieren/normalisieren für jedes Modell andere Zahlen rgb-werte
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Verwendung von methode um die größte Aktivierung zu bestimmen
    targets = None

    # wählt den cam- Algo der in der args steht
    cam_algorithm = methods[args.method]

    #wird freigegeben auch wenn ist in einer äußerenschleife neu erstellt werden muss
    with cam_algorithm(model=model,target_layers=target_layers,use_cuda=args.use_cuda) as cam:
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth, #rote zentrierung größer
                            eigen_smooth=args.eigen_smooth) # gelbe umgebung wird kleiner
        #Extrahiert die Aktivierungskarte des ersten Bildes im Batch  0 für eindimensionalität zum plotten (1Bild = 0)
        grayscale_cam = grayscale_cam[0, :]
        # Aktivierungskarte auf dem Bild
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # konvertieren von rgb auf bgr
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    #backpropagieren modifizierbar
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)

    #berechnung der backpropagation none um backpropagation auf alle Klassen des Modells anzuwenden
    gb = gb_model(input_tensor, target_category=None)

    #Ergebnisse der drei Kanäle wird zusammengeführt
    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])

    #cam und backprop wird multipliziert um eine verknüpfung zu schaffen  und zurück auf das Bild zukonvertieren
    cam_gb = deprocess_image(cam_mask * gb)

    # gb wird auf das bild konvergiert
    gb = deprocess_image(gb)

    #cam vizualisierung wird als datei gespeichert
    cv2.imwrite(f'Grad_cam.jpg', cam_image)
    # gb vizualisierung wird als datei gespeichert
    cv2.imwrite(f'{args.method}_gb.jpg', gb)
    #verknüpfung von cam und gb wird als datei gespeichert
    cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)

