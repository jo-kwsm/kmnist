import torch.nn as nn

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

class KmnistGradCAM:
    def __init__(self, model: nn.Module, model_name: str):
        self.adapt = model_name.startswith("resnet")
        if self.adapt:
            target_layer = model.layer4[1].conv1
            self.gradcam = GradCAM(model, target_layer)
            self.gradcam_pp = GradCAMpp(model, target_layer)


    def __call__(self, x):
        if self.adapt:
            mask, _ = self.gradcam(x)
            heatmap, result = visualize_cam(x)

            mask_pp, _ = self.gradcam_pp(x)
            heatmap_pp, result_pp = visualize_cam(x)

            cam_imgs =[x.cpu(), heatmap, heatmap_pp, result, result_pp]

            return cam_imgs
