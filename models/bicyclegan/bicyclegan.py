import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--data_size", type=int, default=40000, help="The size of data set")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()

learning_rate = 0.0002
img_height = 128
img_width = 128
channels = 3
latent_dim = 8
lambda_pixel = 10
lambda_latent = 0.5
lambda_kl = 0.01

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
print(cuda)

input_shape = (channels, img_height, img_width)

# Loss functions
mae_loss = torch.nn.L1Loss()

# Initialize generator, encoder and discriminators
generator = UnetGenerator(latent_dim, channels, img_height, img_width)
encoder = Encoder(latent_dim, input_shape)
D_VAE = NLayerDiscriminator(input_shape)
D_LR = NLayerDiscriminator(input_shape)

if torch.cuda.is_available():
    generator = generator.cuda()
    encoder.cuda()
    D_VAE = D_VAE.cuda()
    D_LR = D_LR.cuda()
    mae_loss.cuda()

if opt.epoch != 0:
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    encoder.load_state_dict(torch.load("saved_models/%s/encoder_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_VAE.load_state_dict(torch.load("saved_models/%s/D_VAE_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_LR.load_state_dict(torch.load("saved_models/%s/D_LR_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    generator.apply(weights_init_normal)
    D_VAE.apply(weights_init_normal)
    D_LR.apply(weights_init_normal)

# Optimizers
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=learning_rate, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

dataloader = DataLoader(
    ImageDataset("../../dataset/%s" % opt.dataset_name, input_shape, data_size=opt.data_size),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.num_workers,
)
val_dataloader = DataLoader(
    ImageDataset("../../dataset/%s" % opt.dataset_name, input_shape, mode="val", data_size=opt.data_size),
    batch_size=8,
    shuffle=True,
    num_workers=1,
)


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    generator.eval()
    imgs = next(iter(val_dataloader))
    img_samples = None
    for img_A, img_B in zip(imgs["A"], imgs["B"]):
        # Repeat input image by number of desired columns
        real_A = img_A.view(1, *img_A.shape).repeat(latent_dim, 1, 1, 1)
        real_A = Variable(real_A.type(Tensor))
        # Sample latent representations
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (latent_dim, latent_dim))))
        # Generate samples
        fake_B = generator(real_A, sampled_z)
        # Concatenate samples horisontally
        fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
        img_sample = torch.cat((img_A, fake_B), -1)
        img_sample = img_sample.view(1, *img_sample.shape)
        # Concatenate with previous samples vertically
        if img_samples is None:
            img_samples = img_sample
        else:
            img_samples = torch.cat((img_samples, img_sample), -2)
    save_image(img_samples, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=8, normalize=True)
    generator.train()


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z

valid = 1
fake = 0

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        optimizer_E.zero_grad()
        optimizer_G.zero_grad()

        # ----------
        # conditional VAE-GAN: B -> z -> B'
        # ----------

        mu, logvar = encoder(real_B)
        encoded_z = reparameterization(mu, logvar)
        fake_B = generator(real_A, encoded_z)

        loss_pixel = mae_loss(fake_B, real_B)
        loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
        loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)

        sampled_z = Variable(Tensor(np.random.normal(0, 1, (real_A.size(0), latent_dim))))
        _fake_B = generator(real_A, sampled_z)

        loss_LR_GAN = D_LR.compute_loss(_fake_B, valid)

        loss_GE = loss_VAE_GAN + loss_LR_GAN + lambda_pixel * loss_pixel + lambda_kl * loss_kl

        loss_GE.backward(retain_graph=True)
        optimizer_E.step()

        _mu, _ = encoder(_fake_B)
        loss_latent = lambda_latent * mae_loss(_mu, sampled_z)

        loss_latent.backward()
        optimizer_G.step()

        # ----------------------------------
        #  Train Discriminator (cVAE-GAN)
        # ----------------------------------

        optimizer_D_VAE.zero_grad()
        loss_D_VAE = D_VAE.compute_loss(real_B, valid) + D_VAE.compute_loss(fake_B.detach(), fake)
        loss_D_VAE.backward()
        optimizer_D_VAE.step()

        # ---------------------------------
        #  Train Discriminator (cLR-GAN)
        # ---------------------------------

        optimizer_D_LR.zero_grad()
        loss_D_LR = D_LR.compute_loss(real_B, valid) + D_LR.compute_loss(_fake_B.detach(), fake)
        loss_D_LR.backward()
        optimizer_D_LR.step()

        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D VAE_loss: %f, LR_loss: %f] [G loss: %f, pixel: %f, kl: %f, latent: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D_VAE.item(),
                loss_D_LR.item(),
                loss_GE.item(),
                loss_pixel.item(),
                loss_kl.item(),
                loss_latent.item(),
                time_left,
            )
        )

        with open('loss.csv', 'a') as f:
            f.write("%s, %s, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n" % (epoch, i, loss_D_VAE.item(), loss_D_LR.item(), loss_GE.item(), loss_pixel.item(), loss_kl.item(), loss_latent.item()))

        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(encoder.state_dict(), "saved_models/%s/encoder_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_VAE.state_dict(), "saved_models/%s/D_VAE_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_LR.state_dict(), "saved_models/%s/D_LR_%d.pth" % (opt.dataset_name, epoch))
