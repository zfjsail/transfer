
from torch.utils.data import Dataset

from utils import data_utils
from utils import settings


class ProcessedCNNInputDataset(Dataset):

    def __init__(self, entity_type, role):

        data_dir = settings.DOM_ADAPT_DIR
        fname = "{}_{}.pkl".format(entity_type, role)
        data_dict = data_utils.load_large_obj(data_dir, fname)
        self.x1 = data_dict["x1"]
        self.x2 = data_dict["x2"]
        self.y = data_dict["y"]

        self.N = len(self.y)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]

