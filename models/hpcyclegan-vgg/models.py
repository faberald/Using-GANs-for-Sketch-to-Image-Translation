import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
import argparse
import random
import functools
import torch.nn as nn
import torch.utils
from torch.autograd import Variable

from torchvision.models import resnet18
import torchvision.models as models


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, gpu_ids=[]):
        super(UNetDown, self).__init__()
        self.gpu_ids = gpu_ids
        layers = [nn.Conv2d(in_size, out_size, 3, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size, 0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        #     return self.model(input)
        return self.model(x)


# Defines the unet with skip connection.
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class UnetGenerator1(nn.Module):
    def __init__(self, latent_dim, channels, img_height, img_width):
        super(UnetGenerator1, self).__init__()

        channels = channels
        self.height = img_height
        self.width = img_width
        input_nc = channels + 1

        self.fc = nn.Linear(latent_dim, self.height * self.width)

        self.down1 = UNetDown(channels + 1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512, normalize=False)
        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(128, channels, 3, stride=1, padding=1), nn.Tanh()
        )

    def forward(self, x, z):
        # Propogate noise through fc layer and reshape to img shape
        z = self.fc(z).view(z.size(0), 1, self.height, self.width)
        d1 = self.down1(torch.cat((x, z), 1))
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        return self.final(u6)


##############################
#        Encoder
##############################


class Encoder(nn.Module):
    '''
    EncoderLayer
    part of VGG19 (through relu_4_1)
    ref:
    https://arxiv.org/pdf/1703.06868.pdf (sec. 6)
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    '''

    def __init__(self, batch_norm):
        super(Encoder, self).__init__()
        conf = models.vgg.cfgs['E'][:12]  # VGG through relu_4_1
        self.features = models.vgg.make_layers(conf, batch_norm=batch_norm)

    def forward(self, x):
        return self.features(x)

def load_my_state_dict(self, state_dict):
    
    own_state = state_dict
    for name, param in state_dict.items():
        if name not in own_state:
                continue
        # if isinstance(param, Parameter):
        #     # backwards compatibility for serialized parameters
        param = param.data
        own_state[name].copy_(param)

def make_encoder(model_file, batch_norm=True):
    '''
    make a pretrained partial VGG-19 network
    '''
    VGG_TYPE = 'vgg19_bn' if batch_norm else 'vgg19'

    enc = Encoder(batch_norm)

    if model_file and os.path.isfile(model_file):
        # load weights from pre-saved model file
        load_my_state_dict(torch.load(model_file), enc.state_dict())
    else:
        # load weights from pretrained VGG model
        vgg_weights = torch.utils.model_zoo.load_url(models.vgg.model_urls[VGG_TYPE])
        w = {}
        for key in enc.state_dict().keys():
            try:
                w[key] = vgg_weights[key]
            except:
                pass
        load_my_state_dict(w, enc.state_dict())
        if not model_file:
            model_file = "encoder.model"
        torch.save(enc.state_dict(), model_file)

    return enc

def make_decoder(model_file):
    '''
    make a pretrained partial VGG-19 network
    '''

    dec = DecoderLayer()

    if model_file and os.path.isfile(model_file):
        # load weights from pre-saved model file
        load_my_state_dict(torch.load(model_file), enc.state_dict())
    else:
        raise ValueError('Decoder model is not found!')

    return dec

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        target_tensor = target_tensor.cuda()
        return self.loss(input, target_tensor)

class PerceptualLoss(nn.Module):
    '''
    Implement Perceptual Loss in a VGG network

    ref:
    https://arxiv.org/abs/1603.08155

    input: BxCxHxW, BxCxHxW
    output: loss type Variable
    '''

    def __init__(self, vgg_model, n_layers):
        super(PerceptualLoss, self).__init__()
        self.vgg_layers = vgg_model.features

        # use relu_1_1, 2_1, 3_1, 4_1
        if n_layers == 3:
            self.use_layer = set(['2', '25', '29'])
        elif n_layers == 2:
            self.use_layer = set(['2', '25'])
        # self.use_layer = set(['2', '9', '16', '29'])
        self.mse = torch.nn.MSELoss()

    def forward(self, g, s):
        loss = 0

        for name, module in self.vgg_layers._modules.items():

            g, s = module(g), module(s)
            if name in self.use_layer:
                s = Variable(s.data, requires_grad=False)
                loss += self.mse(g, s)

        return loss

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(NLayerDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        channels, _, _ = input_shape
        # Extracts discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs