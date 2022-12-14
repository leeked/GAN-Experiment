import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vis(lossG, lossD):
  """
  Visualization of loss over epochs
  """
  plt.figure()
  plt.subplot(1,1,1)
  plt.plot(np.arange(len(lossG)), lossG, c='r')
  plt.plot(np.arange(len(lossD)), lossD, c='y')
  plt.legend(['Generator','Discriminator'])
  plt.grid(True)
  plt.title('Loss over Epochs')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.savefig("log/loss.png")

"""

Custom Weight Initialization
    From DCGAN paper, taken from PyTorch tutorial

"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)