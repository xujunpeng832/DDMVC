from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch

class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:#view1，2，5
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:#view1，2，5，4
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:#view1，2，5，4,  3
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class handwritten(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'handwritten.mat')['Y'].astype(np.int32).reshape(2000,)
        self.V1 = scipy.io.loadmat(path + 'handwritten.mat')['X'][0][0].astype(np.float32)  #240
        self.V2 = scipy.io.loadmat(path + 'handwritten.mat')['X'][0][1].astype(np.float32)  #76


    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        x1 = self.normalize(self.V1[idx])
        x2 = self.normalize(self.V2[idx])
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

    def normalize(self,x):
        """Normalize"""
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x

class Scene_15(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Scene_15.mat')['Y'].astype(np.int32).reshape(4485,)
        self.V1 = scipy.io.loadmat(path + 'Scene_15.mat')['X'][0][0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Scene_15.mat')['X'][0][1].astype(np.float32)
        # self.V3 = scipy.io.loadmat(path + 'Scene_15.mat')['X'][0][2].astype(np.float32)

    def __len__(self):
        return 4485

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        # x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)],self.Y[idx], torch.from_numpy(np.array(idx)).long()

def load_data(dataset):
    if dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "handwritten":
        dataset = handwritten('./data/')
        dims = [240, 76]  
        view = 2
        data_size = 2000
        class_num = 10
    elif dataset == "Scene_15":
        dataset = Scene_15('./data/')
        dims = [20,59]   
        view = 2           
        data_size = 4485    
        class_num = 15     


    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
