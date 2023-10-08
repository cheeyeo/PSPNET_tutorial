import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50
# For dice loss function
import segmentation_models_pytorch as smp
import numpy as np


# Pyramid Pooling Module
# capturing multi scale features
# Output from PPM with different scales concatenated to provide multi-scale feature pyramid
# Skip connection with original feature map provides rich global contextual prior

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, bin_sizes):
        super(PyramidPoolingModule, self).__init__()

        self.pyramid_pool_layers = nn.ModuleList()

        for bin_sz in bin_sizes:
            ppm = nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_sz),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.pyramid_pool_layers.append(ppm)


    def forward(self, x):
        x_size = x.size()
        out = [x]

        for layer in self.pyramid_pool_layers:
            res = F.interpolate(layer(x), x_size[2:], mode="bilinear", align_corners=True)
            out.append(res)
        
        return torch.cat(out, 1)


class AuxiliaryBranch(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryBranch, self).__init__()

        self.aux = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, x, img_size):
        return F.interpolate(self.aux(x), size=img_size, mode="bilinear", align_corners=False)


class PSPNetLoss(nn.Module):
    def __init__(self, num_classes, aux_weight):
        super(PSPNetLoss, self).__init__()
        self.aux_weight = aux_weight
        self.loss_fn = smp.losses.DiceLoss("multiclass", classes=np.arange(num_classes).tolist(), log_loss=True, smooth=1.0)


    def forward(self, preds, labels):
        if isinstance(preds, dict) == True:
            main_loss = self.loss_fn(preds["main"], labels)
            aux_loss = self.loss_fn(preds["aux"], labels)
            loss = (1 - self.aux_weight) * main_loss + self.aux_weight * aux_loss
        else:
            loss = self.loss_fn(preds, labels)
        
        return loss


class PSPNet(nn.Module):
    def __init__(self, in_channels, num_classes, use_aux=False):
        super(PSPNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_aux = use_aux

        # Backbone layers
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        backbone = resnet50(weights=weights, replace_stride_with_dilation=[False, True, True])
        self.initial = nn.Sequential(*list(backbone.children())[:4])
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Pyramid pooling module
        ppm_in_channels = int(backbone.fc.in_features)
        self.ppm = PyramidPoolingModule(in_channels=ppm_in_channels, out_channels=512, bin_sizes=[1, 2, 3, 6])

        # classifier head
        self.cls = nn.Sequential(
            nn.Conv2d(ppm_in_channels * 2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1)
        )

        # main branch composed of PPM + Classifier
        self.main_branch = nn.Sequential(self.ppm, self.cls)

        # Define aux branch
        if (self.training and self.use_aux):
            self.aux_branch = AuxiliaryBranch(in_channels=(ppm_in_channels  // 2), num_classes=self.num_classes)
    

    def forward(self, x):
        input_size = x.shape[-2:]

        # pass through backbone layers
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)

        # get main branch output
        main_output = self.main_branch(x)
        main_output = F.interpolate(main_output, size=input_size, mode="bilinear")

        if (self.training and self.use_aux):
            aux_output = self.aux_branch(x_aux, input_size)
            output = {
                "aux": aux_output,
                "main": main_output
            }
            return output
        
        return main_output


if __name__ == "__main__":
    ppm_test_input = torch.Tensor(2, 2048, 45, 80)
    model = PyramidPoolingModule(in_channels=2048, out_channels=512, bin_sizes=[1, 2, 3, 6])

    test_output = model(ppm_test_input)
    print("PPM TEST INPUT SHAPE > ", ppm_test_input.shape)
    print("PPM TEST OUTPUT > ", test_output.shape)

    # layer 3 of resnet has 1024 channels output
    # aux_test_input = torch.Tensor(2, 1024, 45, 80)
    # model = AuxiliaryBranch(in_channels=1024, num_classes=3)
    # aux_test_output = model(aux_test_input, img_size=(360, 440))
    # print("Aux test input shape > ", aux_test_input.shape)
    # print("Aux test output shape > ", aux_test_output.shape)