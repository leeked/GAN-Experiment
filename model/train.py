import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.utils as vutils
import torchvision.datasets as Dataset
import torchvision.transforms as transforms

from discriminator import DCGANd
from generator import DCGANg
from utils import DEVICE, weights_init, vis

import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help="filepath of data")
parser.add_argument('epochs', type=int, help="number of epochs to train")
args = parser.parse_args()

def train(
    ganG    : DCGANg,
    ganD    : DCGANd,
    traindl : DataLoader,
    epochs  : int,
    nz      : int,
    lr      : float,
    beta1   : float,
):
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device = DEVICE)

    real_label = 1
    fake_label = 0

    optimizerD = optim.Adam(ganD.parameters(), lr=lr, betas = (beta1, 0.999))
    optimizerG = optim.Adam(ganG.parameters(), lr=lr, betas = (beta1, 0.999))

    # Tracking
    img_list = []
    G_losses = []
    D_losses = []
    print("Starting training loop...")

    for epoch in range(epochs):
        # Tracking
        Gmean_epoch_loss = []
        Dmean_epoch_loss = []

        print(f"EPOCH: {epoch}")

        for img in tqdm(traindl, total=len(traindl)):

            """

            Update Discriminator

            """
            # Train on real batch
            ganD.zero_grad()

            real_cpu = img[0].to(DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)

            output = ganD(real_cpu).view(-1)

            errD_real = criterion(output, label)
            errD_real.backward()

            # Train on fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=DEVICE)

            fake = ganG(noise)
            label.fill_(fake_label)

            output = ganD(fake.detach()).view(-1)

            errD_fake = criterion(output, label)

            errD_fake.backward()

            errD = errD_real + errD_fake

            optimizerD.step()

            """

            Update Generator

            """
            ganG.zero_grad()

            label.fill_(real_label) # Fake labels are real for generator

            output = ganD(fake).view(-1)

            errG = criterion(output, label)

            errG.backward()

            optimizerG.step()

            Gmean_epoch_loss.append(errG.item())
            Dmean_epoch_loss.append(errD.item())
        
        Gmean_loss = np.mean(np.array(Gmean_epoch_loss))
        Dmean_loss = np.mean(np.array(Dmean_epoch_loss))

        print(f" Mean Generator Loss: {Gmean_loss: .4f}\n Mean Discriminator Loss: {Dmean_loss: .4f}")
        G_losses.append(Gmean_loss)
        D_losses.append(Dmean_loss)

        # Check Generator progress
        with torch.no_grad():
            fake = ganG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    vis(G_losses, D_losses)
    return ganG.state_dict(), ganD.state_dict(), img_list


def main():
    # Parse arguments
    print("Initializing...")
    path = args.path
    epochs = args.epochs

    image_size = 64
    lr = 0.0003
    beta1 = 0.5

    nc = 3      # Number of channels (RGB = 3)
    nz = 100    # Size of latent vector (generator input)
    ngf = 64    # Size of feature maps in generator
    ndf = 64    # Size of feature maps in discriminator

    # Prepare data
    print("Preparing data...")
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
    ])

    invNorm = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0.], std = [ 1/0.5, 1/0.5, 1/0.5]),
                                    transforms.Normalize(mean = [ -0.5, -0.5, -0.5], std = [ 1., 1., 1. ]),])

    trainds = Dataset.ImageFolder(root = path, transform = preprocess)

    traindl = DataLoader(trainds, batch_size = 32, num_workers = 2, shuffle = True,)

    # Create models
    print("Preparing models...")
    ganG = DCGANg(nc = nc, nz = nz, ngf = ngf).to(DEVICE)
    ganG.apply(weights_init)

    ganD = DCGANd(nc = nc, nz = nz, ndf = ndf).to(DEVICE)
    ganD.apply(weights_init)
    
    # Enter training
    print("Entering training...")
    resG, resD, img_list = train(
        ganG,
        ganD,
        traindl,
        epochs,
        nz,
        lr,
        beta1,
    )

    # See final Generator results
    print("Saving final images...")
    plt.figure()
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(img_list[-1], (1,2,0)))
    plt.savefig("log/results.png")

    print("Finishing...")



if __name__=="__main__":
    main()