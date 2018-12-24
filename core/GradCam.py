import cv2
import numpy as np
import torch

from torch.nn import ReLU
from torch.autograd import Variable
from .Base import Base


from .utils import tensor2cam

class GradCam(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handles = []
        self.gradients = None
        self.conv_outputs = None

    def store_outputs_and_grad(self, layer):
        def store_grad(grad):
            self.gradients = grad

        def store_outputs(module, input, outputs):
            if module == layer:
                self.conv_outputs = outputs
                self.handles.append(outputs.register_hook(store_grad))

        self.handles.append(layer.register_forward_hook(store_outputs))

    def guide(self, module):
        def guide_relu(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0.0),)

        for module in module.modules():
            if isinstance(module, ReLU):
                self.handles.append(module.register_backward_hook(guide_relu))

    def clean(self):
        [h.remove() for h in self.handles]

    def __call__(self, input_image, layer, guide=False, target_class=None, postprocessing=lambda x: x):
        self.clean()
        self.store_outputs_and_grad(layer)
        if guide: self.guide(self.module)

        input_var = Variable(input_image, requires_grad=True).to(self.device)
        predictions = self.module(input_var)

        if target_class == None: _, target_class = torch.max(predictions, dim=1)

        print(target_class)

        target = torch.zeros(predictions.size()).to(self.device)
        target[0][target_class] = 1

        self.module.zero_grad()
        predictions.backward(gradient=target, retain_graph=True)

        with torch.no_grad():
            avg_channel_grad = self.gradients.data.squeeze().mean(1).mean(1)
            outputs = self.conv_outputs
            b, c, w, h = outputs.shape

            cam = avg_channel_grad @ outputs.view((c, w * h))
            cam = cam.view(h, w)
            with torch.no_grad():
                image_with_heatmap = tensor2cam(postprocessing(input_image.squeeze()), cam)

        self.clean()

        return image_with_heatmap.unsqueeze(0)


