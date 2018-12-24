import torch

from core import *
from core.utils import image_net_postprocessing
from PIL import Image

from torchvision.models import alexnet, vgg16, resnet18
from torchvision.transforms import ToTensor, Resize, Compose

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# create a model
model = alexnet(pretrained=True)
print(model)
cat = Image.open("/Users/vaevictis/Documents/Project/A-journey-into-Convolutional-Neural-Network-visualization-/images/cat.jpg")
# resize the image and make it a tensor
input = Compose([Resize((224,224)), ToTensor()])(cat)
# add 1 dim for batch
input = input.unsqueeze(0)
# call mirror with the input and the model
layers = list(model.children())
layer = layers[0][12]
print(layer)

def imshow(tensor):
    tensor = tensor.squeeze()
    print(tensor.shape)
    if len(tensor.shape) > 2: tensor = tensor.permute(1, 2, 0)
    img = tensor.cpu().numpy()
    plt.imshow(img, cmap='gray')
    plt.show()


vis = GradCam(model.to(device), device)
img = vis(input.to(device), layer, target_class=285, postprocessing=image_net_postprocessing, guide=True)

print(img.shape)
with torch.no_grad():
    imshow(img[0])