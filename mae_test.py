import torch
import argparse
from PIL import Image 
from torchvision.transforms import transforms
from torch.cuda.amp import autocast as autocast
import numpy as np 
from model.Transformers.VIT.mae import MAEVisionTransformers as MAE
from loss.mae_loss import build_mask


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='pretrained_mae_ckpts/vit-mae_losses_0.20102281799793242.pth', type=str)
# parser.add_argument('--test_image', default='/workspace/data/food/classification/KFOOD201.classification/val/000/Img_116_1124.jpg', type=str)
# parser.add_argument('--test_image', default='/workspace/data/food/classification/KFOOD201.classification/val/001/Img_067_0964.jpg', type=str)
# parser.add_argument('--test_image', default='/workspace/data/food/classification/KFOOD201.classification/val/002/Img_110_0956.jpg', type=str)
# parser.add_argument('--test_image', default='test_images/stock-image-praying-mantis.jpg', type=str)
# parser.add_argument('--test_image', default='test_images/cats_dogs.jpg', type=str)
# parser.add_argument('--test_image', default='test_images/cats_dogs1.jpg', type=str)
parser.add_argument('--test_image', default='test_images/shutterstock_795502450.jpg', type=str)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--patch_size', default=8, type=int)

args = parser.parse_args()


image = Image.open(args.test_image)
raw_image = image.resize((args.crop_size, args.crop_size))
raw_image.save("output/src_image.jpg")
raw_tensor  = torch.from_numpy(np.array(raw_image))
print(raw_tensor.shape)


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

image = transforms.Compose([
    transforms.Resize((args.crop_size, args.crop_size)),
    # transforms.CenterCrop(args.crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)]
)(image)
image_tensor = image.unsqueeze(0)

ckpt = torch.load(args.ckpt, map_location="cpu")['state_dict']

model = MAE(
    img_size = args.crop_size,
    patch_size = args.patch_size,  
    encoder_dim = 768,
    encoder_depth = 12,
    encoder_heads = 12,
    decoder_dim = 512,
    decoder_depth = 8,
    decoder_heads = 16, 
    mask_ratio = 0.75
)


print(model)
model.load_state_dict(ckpt, strict=True)
model.cuda()
model.eval()
image_tensor = image_tensor.cuda()
with torch.no_grad():
    with autocast():
        output, mask_index = model(image_tensor)
        print(output.shape)
        
output_image = output.squeeze(0)
output_image = output_image.permute(1,2,0).cpu().numpy()
output_image = output_image * std + mean
output_image = output_image * 255

import cv2 
output_image = output_image[:,:,::-1]
cv2.imwrite("output/output_image.jpg", output_image)


mask_map = build_mask(mask_index, patch_size=args.patch_size, img_size=args.crop_size)

non_mask = 1 - mask_map 
non_mask = non_mask.unsqueeze(-1)

non_mask_image = non_mask * raw_tensor


mask_map = mask_map * 127
mask_map = mask_map.unsqueeze(-1)

print(torch.min(mask_map))

non_mask_image  += mask_map 

# print(non_mask_image)
non_mask_image = non_mask_image.cpu().numpy()
print(non_mask_image.shape)
cv2.imwrite("output/mask_image.jpg", non_mask_image[:,:,::-1])

print(output_image)
    
        
        
        
        

        