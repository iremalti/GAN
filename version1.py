model = netD
model.eval()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.modules.conv.Conv2d):
        target_layer = module
    else:
        continue
print(target_layer)
rgb_img = cv2.imread('C:\\Users\\altipair\\Desktop\\bachelor\\ausgabe\\fake_sample_bild_{}.png'.format(epoch), 1)[:, :,
          ::-1]
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
targets = None

cam = GradCAM(model, target_layer, use_cuda=False)
cam.batch_size = 32
grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

output_filename = os.path.join(output_folder, 'Bild_{}_cam.jpg'.format(epoch))
cv2.imwrite(output_filename, cam_image)