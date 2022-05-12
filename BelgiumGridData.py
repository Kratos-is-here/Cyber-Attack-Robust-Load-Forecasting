import torch
import pandas as pd
import numpy as np
import time as t
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  

class load_data(DataLoader):
      def __init__(self, csv_file, transform=None,attack='scaling', testing = False, x = 100000):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.testing = testing
        self.dataset = pd.read_csv(csv_file)
        self.transform = transform
        self.attack=attack
      def __len__(self):
        return len(self.dataset)

      def __getitem__(self, idx):
        # if self.attack=='scaling':
        #     c=5
        # if self.attack=='ramping':
        #     c=7
        # if self.attack=='random':
        #     c=9

        data = self.dataset.iloc[idx,1]
        label = self.dataset.iloc[idx,3]
        t.sleep(x)

        if self.testing:
          return data
        
        return (data,label)
