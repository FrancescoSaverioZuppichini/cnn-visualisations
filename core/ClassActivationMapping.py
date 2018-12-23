import torch

from torch.nn import AvgPool2d, Conv2d, Linear, ReLU
from torch.nn.functional import softmax

from .Base import Base

from .utils import module2traced, imshow, tensor2cam



class ClassActivationMapping(Base):
    """
    Based on Learning Deep Features for Discriminative Localization (https://arxiv.org/abs/1512.04150).
    Be aware,it requires feature maps to directly precede softmax layers.
    It will work for resnet but not for alexnet for example
    """

    def guide(self, module):
        def guide_relu(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0.0),)

        for module in module.modules():
            if isinstance(module, ReLU):
                self.handles.append(module.register_backward_hook(guide_relu))

    def __call__(self, inputs, layer, target_class=285, postprocessing=lambda x: x, guide=False):
        modules = module2traced(self.module, inputs)
        last_conv = None
        last_linear = None
        if guide: self.guide(self.module)

        for i, module in enumerate(modules):
            if isinstance(module, Conv2d):
                last_conv = module
            if isinstance(module, AvgPool2d):
                pass
            if isinstance(module, Linear):
                last_linear = module
                print(i, module)

        def store_conv_outputs(module, inputs, outputs):
            self.conv_outputs = outputs

        last_conv.register_forward_hook(store_conv_outputs)

        predictions = self.module(inputs)

        if target_class == None: _, target_class = torch.max(predictions, dim=1)

        _, c, h, w = self.conv_outputs.shape
        # get the weights relative to the target class
        fc_weights_class = last_linear.weight.data[target_class]
        # sum upp the multiplication of each weight w_k for the relative channel in the last
        # convolution output
        cam = fc_weights_class @ self.conv_outputs.view((c, h * w))
        cam = cam.view(h, w)

        with torch.no_grad():
            image_with_heatmap = tensor2cam(postprocessing(inputs.squeeze()), cam)

        return image_with_heatmap.unsqueeze(0)