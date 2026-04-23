from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from tqdm import tqdm
import time

class SmartDetectorDataset(Dataset):
    def __init__(self, X, y,processor, protocols = None):
        self.X = X
        self.y = y
        self.protocols = protocols
        self.processor = processor
        self.features = None
        batch_lists = []
        batch_size = 512
        print('Construce datasets, calculating positions...')
        start = time.perf_counter()
        for i in range(0,X.shape[0],batch_size):
            batch_flow = self.processor.get_sam(X[i:i+batch_size],only_pos=True,give_pos = False)
            batch_lists.append(batch_flow)
        end = time.perf_counter()
        print(f'Finsh calculating positions after {end-start}s.')
        self.posX = torch.cat(batch_lists,dim=0)
        
    def cache_features(self, model):
        batch_size = 512
        batch_lists = []
        model.only_extract_feature = True
        with torch.no_grad():
            for i in range(0,self.X.shape[0],batch_size):
                batch_sam = self.processor.get_sam(self.X[i:i+batch_size],
                                                    only_pos=False,give_pos = True,
                                                    position=self.posX[i:i+batch_size]).to('cuda')
                batch_features = model(batch_sam)
                batch_lists.append(batch_features)
            self.features = torch.cat(batch_lists,dim=0).to('cpu')
        model.only_extract_feature = False

        
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
        if self.features is not None:
            return self.X[idx],self.posX[idx], self.features[idx], self.y[idx]
        else:
            return self.X[idx],self.posX[idx], self.y[idx]


if __name__=='__main__':
    print(6)