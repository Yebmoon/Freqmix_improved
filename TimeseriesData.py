from torch.utils.data import DataLoader, Dataset,random_split
import torch
import random

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y,max_len):
        self.X = [torch.tensor(seq, dtype=torch.float32) for seq in X]
        self.y = torch.tensor(y,dtype=torch.long)
        self.max_len = max_len

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
        original_data = self.X[idx]
        original_label = self.y[idx]
        seq, length = self.pad_or_truncate(original_data)        
        return seq,original_label,length
        