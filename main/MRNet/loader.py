import numpy as np
import os
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data

from torch.autograd import Variable

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

class Dataset(data.Dataset):
    def __init__(self, datadir, diagnosis, use_gpu):
        super().__init__()
        self.use_gpu = use_gpu

        if ("train" in datadir[0]) or ("valid" in datadir[0]):
          labels = pd.read_csv("/content/drive/MyDrive/Spring 2022/BMI 260/BMI260_project/MRNet-v1.0/train-acl.csv", index_col = 0, header = None, names=["patient", "acl_tear"])
        else:
          labels = pd.read_csv("/content/drive/MyDrive/Spring 2022/BMI 260/BMI260_project/MRNet-v1.0/test-acl.csv", index_col=0, header=None, names=["patient", "acl_tear"])

        data_labels = []
        self.paths = []
        for file in os.listdir(datadir[0]):
          sample = int(file.strip(".npy"))
          label = int(labels.loc[sample]["acl_tear"])
          self.paths.append(datadir[0] + "/" + file)
          data_labels.append((datadir[0] + "/" + file, label))
        data_labels = np.asarray(data_labels, dtype=object)
        self.labels = data_labels[:,1]
        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        return loss

    def __getitem__(self, index):
        path = self.paths[index]
        with open(path, 'rb') as file_handler: # Must use 'rb' as the data is binary
            vol = np.load(file_handler).astype(np.int32)
            # vol = pickle.load(file_handler).astype(np.int32)

        # crop middle
        pad = int((vol.shape[2] - INPUT_DIM)/2)
        vol = vol[:,pad:-pad,pad:-pad]
        
        # standardize
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL

        # normalize
        vol = (vol - MEAN) / STDDEV
        
        # print("post crop", vol.shape)
        # convert to RGB
        vol = np.stack((vol,)*3, axis=1)
        # print("post rgb", vol.shape)
        vol_tensor = torch.FloatTensor(vol)
        label_tensor = torch.FloatTensor([self.labels[index]])

        return vol_tensor, label_tensor

    def __len__(self):
        return len(self.paths)

def load_data(diagnosis, plane, use_gpu=False):
    train_dir = ["/content/drive/MyDrive/Spring 2022/BMI 260/BMI260_project/MRNet-v1.0/train/"+plane]
    valid_dir = ["/content/drive/MyDrive/Spring 2022/BMI 260/BMI260_project/MRNet-v1.0/valid/"+plane]
    test_dir = ["/content/drive/MyDrive/Spring 2022/BMI 260/BMI260_project/MRNet-v1.0/test/"+plane]

    #train_dir = ["/content/drive/MyDrive/CS235/Project/BMI260/MRNet-v1.0/train/"+plane]
    #valid_dir = ["/content/drive/MyDrive/CS235/Project/BMI260/MRNet-v1.0/valid/"+plane]
    #test_dir = ["/content/drive/MyDrive/CS235/Project/BMI260/MRNet-v1.0/test/"+plane]
    
    train_dataset = Dataset(train_dir, diagnosis, use_gpu)
    valid_dataset = Dataset(valid_dir, diagnosis, use_gpu)
    test_dataset = Dataset(test_dir, diagnosis, use_gpu)

    train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=8, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=8, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False)

    return train_loader, valid_loader, test_loader
