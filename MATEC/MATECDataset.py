from torch.utils.data import DataLoader, Dataset
import torch

class MATECDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)

    def pad_or_truncate(self,sequence):
        original_length = sequence.size(0)
        if original_length>self.max_len:
            sequence = sequence[:self.max_len]
            return sequence, self.max_len
        else:
            sequence = torch.cat((sequence,torch.zeros(self.max_len-original_length)))
            return sequence,original_length


    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__=='__main__':
    print(6)