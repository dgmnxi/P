import torch
from torch import nn
from typing import Optional, Callable, List, Type
from torchvision import models

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Residual Connection
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,
                 block: Type[BasicBlock],
                 layers: List[int],
                 output_dim: int = 128,
                 zero_init_residual: bool = False,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1

        # --- Input layer modified for 1-channel spectrogram ---
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- Output layer modified for 128-dim vector ---
        self.fc = nn.Linear(512 * block.expansion, output_dim)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18(**kwargs) -> ResNet:
    """Constructs a ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

# 기학습된 파라미터를 통해 전이학습
def resnet18_transfer_learning(output_dim: int = 128, freeze_features: bool = True) -> ResNet:

    # 1. Load pretrained ResNet-18 model
    pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 2. Instantiate our custom ResNet-18 structure
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_dim=output_dim)
    
    # 3. Copy weights from the pretrained model, skipping mismatched layers
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()
    
    # Filter out weights that don't match in shape (conv1 and fc)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    
    # Update the current model's state dict with the pretrained weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # 4. Freeze layers if requested
    if freeze_features:
        for name, param in model.named_parameters():
            # Unfreeze the first conv layer and the final fc layer
            if not (name.startswith('conv1') or name.startswith('fc')):
                param.requires_grad = False
                
    # Verify which layers are frozen
    print("--- Layer Freezing Status ---")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    print("-----------------------------")
        
    return model

# Example of how to use it:
if __name__ == '__main__':
    # Create a dummy input tensor of shape (batch_size, channels, height, width)
    # This matches the output of your mel_spectrogram
    dummy_input = torch.randn(1, 1, 128, 431)

    # Instantiate the model from scratch
    print("--- Model from Scratch ---")
    model = resnet18(output_dim=128)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # --- Transfer Learning Example ---
    print("\n--- Transfer Learning Model ---")
    # Instantiate the transfer learning model
    transfer_model = resnet18_transfer_learning(output_dim=128, freeze_features=True)
    
    # Get the output
    transfer_output = transfer_model(dummy_input)
    
    # Print the output shape
    print(f"Input shape: {dummy_input.shape}")
    print(f"Transfer model output shape: {transfer_output.shape}")

