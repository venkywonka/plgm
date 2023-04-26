
# note that this custom dataset is not prepared on the top of geometric Dataset(pytorch's inbuilt)
import os
import torch
import glob
import numpy as np
import random
import math
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset as Dataset_n
from torch_geometric.data import DataLoader as DataLoader_n

class LabelledDataset(Dataset_n):
    def __init__(self, npy_file, processed_dir):
      self.npy_ar = np.load(npy_file)
      self.processed_dir = processed_dir
      self.protein_1 = self.npy_ar[:,2]
      self.protein_2 = self.npy_ar[:,5]
      self.label = self.npy_ar[:,6].astype(float)
      self.n_samples = self.npy_ar.shape[0]

    def __len__(self):
      return(self.n_samples)

    def __getitem__(self, index):
      prot_1 = os.path.join(self.processed_dir, self.protein_1[index]+".pt")
      prot_2 = os.path.join(self.processed_dir, self.protein_2[index]+".pt")
      #print(f'Second prot is {prot_2}')
      try:
        prot_1 = torch.load(glob.glob(prot_1)[0])
        #print(f'Here lies {glob.glob(prot_2)}')
        prot_2 = torch.load(glob.glob(prot_2)[0])
      except:
        print("Error in loading")
        print(prot_1)
        print(prot_2)
        return None, None, None
      return prot_1, prot_2, torch.tensor(self.label[index])

def custom_collate_fn(batch):
    filtered_batch = [item for item in batch if item[0] is not None and item[1] is not None and item[2] is not None]
    prot_1_batch = torch.stack([item[0] for item in filtered_batch])
    prot_2_batch = torch.stack([item[1] for item in filtered_batch])
    labels_batch = torch.stack([item[2] for item in filtered_batch])
    return prot_1_batch, prot_2_batch, labels_batch


def prepare_data(processed_dir="/nethome/vganesh41/worktrees/plgm/main/plgm/Human_features/", npy_file = "/nethome/vganesh41/worktrees/plgm/main/plgm/Human_features/npy_file_new(human_dataset).npy"):
  dataset = LabelledDataset(npy_file = npy_file ,processed_dir= processed_dir)
  final_pairs =  np.load(npy_file)
  size = final_pairs.shape[0]
  print("Size is : ")
  print(size)
  seed = 42
  torch.manual_seed(seed)
  #print(math.floor(0.8 * size))
  #Make iterables using dataloader class
  trainset, testset = torch.utils.data.random_split(dataset, [math.floor(0.8 * size), size - math.floor(0.8 * size) ])
  #print(trainset[0])
  trainloader = DataLoader_n(dataset= trainset, batch_size= 4, num_workers = 0, collate_fn=custom_collate_fn)
  testloader = DataLoader_n(dataset= testset, batch_size= 4, num_workers = 0, collate_fn=custom_collate_fn)
  print("Length")
  print(len(trainloader))
  print(len(testloader))
  return dataset, trainloader, testloader
