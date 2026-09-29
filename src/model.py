"""Unified factory for scratch and pretrained image classifiers."""
from torch import nn
from torchvision import models as tv
from .models.alex_net import AlexNet
from .models.google_net import GoogLeNet
from .models.res_net import BasicBlock, ResNet
from .models.vgg_net import VGG

VGG16_CONFIG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                512, 512, 512, 'M', 512, 512, 512, 'M']


class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7)), nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
                                        nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def create_model(name, num_classes=2, pretrained=False, mode='full'):
    """Build alexnet, vgg16, googlenet, resnet18, or custom_cnn.

    Scratch models come from src/models. Pretrained models use torchvision so
    official ImageNet weights load. mode: frozen, partial, or full.
    """
    key = name.lower().replace('-', '').replace('_', '')
    key = {'alex': 'alexnet', 'vgg': 'vgg16', 'googlenetv1': 'googlenet',
           'resnet': 'resnet18'}.get(key, key)
    if mode not in {'frozen', 'partial', 'full'}:
        raise ValueError('mode must be frozen, partial, or full')
    if key == 'customcnn':
        if pretrained or mode != 'full':
            raise ValueError('custom_cnn supports scratch/full only')
        return CustomCNN(num_classes)
    if key not in {'alexnet', 'vgg16', 'googlenet', 'resnet18'}:
        raise ValueError(f'Unknown model: {name}')
    if not pretrained:
        if mode != 'full':
            raise ValueError('Frozen/partial require pretrained=True')
        factories = {
            'alexnet': lambda: AlexNet(num_classes),
            'vgg16': lambda: VGG(VGG16_CONFIG, batch_norm=False, num_classes=num_classes),
            'googlenet': lambda: GoogLeNet(num_classes=num_classes, aux_logits=False),
            'resnet18': lambda: ResNet(BasicBlock, [2, 2, 2, 2], num_classes),
        }
        return factories[key]()
    builders = {
        'alexnet': (tv.alexnet, tv.AlexNet_Weights.DEFAULT),
        'vgg16': (tv.vgg16, tv.VGG16_Weights.DEFAULT),
        'googlenet': (tv.googlenet, tv.GoogLeNet_Weights.DEFAULT),
        'resnet18': (tv.resnet18, tv.ResNet18_Weights.DEFAULT),
    }
    builder, weights = builders[key]
    kwargs = {'weights': weights}
    if key == 'googlenet':
        kwargs['aux_logits'] = False
    model = builder(**kwargs)
    if key in {'alexnet', 'vgg16'}:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        head = 'classifier.6.'
        last_stage = 'features.10.' if key == 'alexnet' else 'features.28.'
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head = 'fc.'
        last_stage = 'inception5b.' if key == 'googlenet' else 'layer4.'
    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad = (mode == 'full' or parameter_name.startswith(head) or
                                   (mode == 'partial' and parameter_name.startswith(last_stage)))
    return model


def make_resnet18(num_classes=2, mode='frozen'):
    """Compatibility alias used by earlier experiments."""
    return create_model('resnet18', num_classes, pretrained=True, mode=mode)
