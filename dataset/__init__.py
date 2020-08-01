import numpy as np
import torch
from torch.utils.data import Dataset

from utils import data_utils
from utils import settings


class ProcessedCNNInputDataset(Dataset):

    def __init__(self, entity_type, role):

        data_dir = settings.DOM_ADAPT_DIR
        fname = "{}_{}.pkl".format(entity_type, role)
        data_dict = data_utils.load_large_obj(data_dir, fname)
        self.x1 = np.array(data_dict["x1"], dtype="float32")
        self.x2 = np.array(data_dict["x2"], dtype="float32")
        self.y = data_dict["y"]

        self.N = len(self.y)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]


class OAGDomainDataset(Dataset):
    ''' Load svmlight-formated datasets, using domain as labels '''
    def __init__(self, entity_type, role, domain=0):

        data_dir = settings.DOM_ADAPT_DIR
        fname = "{}_{}.pkl".format(entity_type, role)
        data_dict = data_utils.load_large_obj(data_dir, fname)
        self.x1 = np.array(data_dict["x1"], dtype="float32")
        self.x2 = np.array(data_dict["x2"], dtype="float32")
        self.x1 = torch.from_numpy(self.x1)
        self.x2 = torch.from_numpy(self.x2)

        self.y = torch.LongTensor([domain] * self.x1.shape[0])

        self.N = self.y.size()[0]

    def __getitem__(self, index):
        return self.x1[index], self.x2[index], self.y[index]

    def __len__(self):
        return self.x1.shape[0]
