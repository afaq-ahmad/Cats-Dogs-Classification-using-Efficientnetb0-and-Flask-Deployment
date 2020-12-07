import argparse
import torch
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image
import numpy as np


parser = argparse.ArgumentParser(description='Model Testing on single image')
parser.add_argument('-wp', '--weight_path', default='weight/model_b.pth.tar',help='Weight Path (default: weight/model_best.pth.tar)')
parser.add_argument('-im', '--image_path', help='Testing Image Path')
args = parser.parse_args()

classes=['cats', 'dogs']
testimg_path=args.image_path
weight_path=args.weight_path

model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=2);
if torch.cuda.is_available():
    gpu=0
else:
    gpu=None

    
if gpu is None:
    checkpoint = torch.load(weight_path,map_location='cpu')
else:
    # Map model to be loaded to specified single gpu.
    loc = 'cuda:{}'.format(gpu)
    checkpoint = torch.load(weight_path, map_location=loc);
best_acc1 = checkpoint['best_acc1']

if type(best_acc1)!=torch.Tensor:
    best_acc1=torch.tensor(best_acc1)
if gpu is not None:
    # best_acc1 may be from a checkpoint from a different GPU
    best_acc1 = best_acc1.to(gpu)
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
model.load_state_dict(checkpoint['state_dict']);

## Same image transformation as in the training time
Test_transformer=transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor()])
img_pil=Image.open(testimg_path)
img_t=Test_transformer(img_pil).float()
img_t=img_t.reshape([1]+list(img_t.shape))

model.eval();
with torch.no_grad():
    if gpu is not None:
        img_t = img_t.cuda(gpu, non_blocking=True)

    # compute output
    output = model(img_t)

    predicted_class=classes[np.argmax(output.tolist(),axis=1)[0]]
    print('Predicted Category: ',predicted_class)
    font=ImageFont.truetype('efficientnet_pytorch/SansSerif.ttf', size=40)
    ImageDraw.Draw(img_pil).text((0, 0),predicted_class,(255, 0, 0),font=font)
    img_pil.show()