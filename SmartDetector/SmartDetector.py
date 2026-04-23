import torch
import torch.nn as nn
import numpy as np
from gensim.models import Word2Vec
from bisect import bisect_left
import random
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import logging
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import os
from SmartDetectorDataset import SmartDetectorDataset

MIXUP = False
PRE_AUG = False
SINGLE_AUG = False
EARLY_STOP_EPOCH = 20
flow_column_types = {
    'start_time': str,
    'flow_id':str,
    'src':str,
    'dst':str,
    'sport':'Int32',
    'dport':'Int32',
    'protocol':'Int32',
    'packets index':str,
    'len_sequences':str,
    'time_interval_sequences':str,
    'label':'Int32'
}
class SAMProcessor:
    def __init__(self, K=40, B=100,seed=42):
        self.K = K
        self.B = B
        self.length_model = None
        self.iat_model = None
        self.seed = seed
        
        self.device = 'cpu'

        self.len_keys = None
        self.iat_keys = None
        
        self.len_emb = None
        self.iat_emb = None
        
        self.len_value_to_idx = None
        self.iat_value_to_idx = None

    def train_embeddings(self, background_flows, model_dir, force_training=False):
        
        lengths = [[l for l in flow] for flow in background_flows[:,0,:]]
        iats = [[iat for iat in flow] for flow in background_flows[:,1,:]]

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if not force_training and os.path.exists(model_dir+'/length_wv.model'):
            print('Length model exists, we will load it without traning.')
            self.length_model = Word2Vec.load(model_dir+'/length_wv.model')
        else:
            print('Now training length model Word2Vec...')
            self.length_model = Word2Vec(sentences=lengths,vector_size=self.B,window=5,min_count=1,sg=0,workers=12,seed=self.seed)
            self.length_model.save(model_dir+'/length_wv.model')
            print('Finish.')

        if not force_training and os.path.exists(model_dir+'/iat_wv.model'):
            print('Iat model exists, we will load it without traning.')
            self.iat_model = Word2Vec.load(model_dir+'/iat_wv.model')
        else:
            print('Now training iat model Word2Vec...')
            self.iat_model = Word2Vec(sentences=iats,vector_size=self.B,window=5,min_count=1,sg=0,workers=12,seed=self.seed)
            self.iat_model.save(model_dir+'/iat_wv.model')
            print('Finish.')

        
        def build_mapping(model_wv):
            
            raw_keys = [k for k in model_wv.index_to_key]
            sorted_indices = np.argsort(raw_keys)
            
            sorted_keys = np.array(raw_keys)[sorted_indices]
            sorted_vectors = model_wv.vectors[sorted_indices]
            
            
            keys_t = torch.tensor(sorted_keys, dtype=torch.float32).to(self.device)
            
            emb_layer = nn.Embedding.from_pretrained(
                torch.tensor(sorted_vectors, dtype=torch.float32), 
                freeze=True
            ).to(self.device)
            
            value_to_idx = {float(v): i for i, v in enumerate(sorted_keys)}
            
            return keys_t, emb_layer,value_to_idx
        
        self.len_keys, self.len_emb,self.len_value_to_idx = build_mapping(self.length_model.wv)
        self.iat_keys, self.iat_emb,self.iat_value_to_idx = build_mapping(self.iat_model.wv)
        
        
        
    
    @torch.no_grad()
    def get_sam(self, flow, only_pos = False, give_pos = False, position =None):
        '''
        flow: torch.Tensor [B, 3, K] (GPU/CPU)
        return: sam [B, 3, K, B] (GPU)
        '''
        Batch_size, _, K = flow.shape
        device = self.device

        
        flow = flow.to(device)
        len_keys = self.len_keys.to(device)
        iat_keys = self.iat_keys.to(device)

        
        sam = torch.zeros(Batch_size, 3, K, self.B, device=device)

        
        if not only_pos:
            sam[:, 1,:,:] = flow[:, 1:2, :].expand(-1, -1, self.B)
        if give_pos:
            positions = position.to(device)
            sam[:, 0,:,:] = self.len_emb(positions[:,0,:])
            sam[:, 2,:,:] = self.iat_emb(positions[:,1,:])
            return sam

        
        pkt_len = flow[:, 0, :]
        pos = torch.searchsorted(len_keys, pkt_len)
        pos = torch.clamp(pos, 1, len(len_keys) - 1)
        left = len_keys[pos - 1]
        right = len_keys[pos]
        closest_idx = torch.where(torch.abs(pkt_len - left) <= torch.abs(pkt_len - right), pos-1, pos)
        if not only_pos:
            emb = self.len_emb(closest_idx)
            sam[:, 0,:,:] = emb
        else:
            len_idx = closest_idx

        
        iat = flow[:, 2, :]
        pos = torch.searchsorted(iat_keys, iat)
        pos = torch.clamp(pos, 1, len(iat_keys) - 1)
        left = iat_keys[pos - 1]
        right = iat_keys[pos]
        closest_idx = torch.where(torch.abs(iat - left) <= torch.abs(iat - right), pos-1, pos)
        if not only_pos:
            emb = self.iat_emb(closest_idx)
            sam[:, 2,:,:] = emb
        else:
            iat_idx = closest_idx

        if only_pos:
            return torch.stack([len_idx,iat_idx],dim=1).to('cpu')    
        else:
            return sam
    
class SmartDetectorEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SmartDetectorEncoder, self).__init__()
        
        self.backbone = resnet50(pretrained=False)
        
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() 
              
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self,x):
        x = x.reshape(x.size(0), 1, -1, x.size(-1))
        h = self.backbone(x)
        z = self.projection_head(h)
        return h, z


def generate_data(n=10, max_len=40, seed=42):
    rng = np.random.default_rng(seed)

    
    lengths = rng.integers(1, max_len + 1, size=n)

    
    X = np.zeros((n, 3, max_len), dtype=float)

    for i in range(n):
        L = lengths[i] 
        X[i, 0, :L] = rng.integers(60, 1501, size=L)
        X[i, 1, :L] = rng.choice([-1, 1], size=L)
        X[i, 2, :L] = rng.random(size=L)

    return X, lengths

def traffic_augmentation(flows,q=0.3,r=0.3):
    B,C,L = flows.shape
    device = flows.device
    dtype = flows.dtype

    real_mask = flows[:, 0, :] > 0 

    flows_aug = flows.clone()
    r_prime = torch.rand(B,L,device=device)
    delay_mask = (r_prime <= r) & real_mask
    deltas = torch.rand(B,L) * 0.2
    flows_aug[:, 2, :] += delay_mask * deltas

    dummy_len = torch.randint(1, 1501, (B, L),device=device,dtype=dtype)
    
    dummy_dir = (torch.randint(0, 2, (B, L), device=device, dtype=dtype) * 2 - 1).to(dtype)
    dummy_iat = torch.rand(B, L, device=device) * 0.2
    dummies = torch.stack([dummy_len, dummy_dir, dummy_iat], dim=1)

    
    combined = torch.zeros((B, 3, 2 * L),device=device,dtype=dtype)
    combined[:, :, 0::2] = dummies  
    combined[:, :, 1::2] = flows_aug    

    
    q_prime = torch.rand(B, L,device=device)
    is_dummy_valid = (q_prime <= q) & real_mask  

    
    mask = torch.ones((B, 2 * L),device=device, dtype=torch.bool)
    mask[:, 0::2] = is_dummy_valid
    mask[:,1::2] = real_mask
    combined = combined * mask.unsqueeze(1)

    sort_idx = torch.argsort(~mask, dim=1,stable=True)

    sort_idx_expanded = sort_idx.unsqueeze(1).expand(-1, 3, -1) 
    augmented = torch.gather(combined, 2, sort_idx_expanded)
    augmented = augmented[:, :, :L]
    return augmented
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temp = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = self.cosine_sim(representations.unsqueeze(1), representations.unsqueeze(0))

        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temp)
        denominator = torch.sum(torch.exp(similarity_matrix / self.temp), dim=-1) - 1

        loss = -torch.log(nominator / denominator).mean()
        return loss

class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(Classifier, self).__init__()
        self.encoder = encoder.backbone 
        self.fc = nn.Linear(2048, num_classes) 
        self.only_extract_feature = False
        self.only_classifier = False

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False


    def forward(self, x):
        if self.only_extract_feature:
            with torch.no_grad():
                x = x.reshape(x.size(0), 1, -1, x.size(-1))
                feature = self.encoder(x)
                return feature
        elif self.only_classifier:
            return self.fc(x)
        else:       
            with torch.no_grad():
                x = x.reshape(x.size(0), 1, -1, x.size(-1))
                feature = self.encoder(x)
            return self.fc(feature)


def save_checkpoint(model,optimizer,scheduler,epoch,loss,best_val_loss,filepath):
    checkpoint = {
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'scheduler_state_dict':scheduler.state_dict() if scheduler is not None else None,
        'loss':loss,
        'best_val_loss':best_val_loss,
        'random_state': random.getstate(),  
        'torch_random_state': torch.get_rng_state(),  
    }
    torch.save(checkpoint,filepath)

def load_checkpoint(filepath,model,optimizer,scheduler):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if (checkpoint.get('scheduler_state_dict',None) is not None) and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    best_val_loss = checkpoint['best_val_loss']
    random.setstate(checkpoint['random_state'])
    torch.set_rng_state(checkpoint['torch_random_state'])
    return model, optimizer,scheduler, epoch, loss,best_val_loss

def pre_aug(x,lengths,strength = 0.2):
    
    
    batch_size = x.shape[0]
    length = x.shape[1]
    device = x.device
    k = (lengths.float() * strength).floor().long()

    mask = torch.arange(length, device=device).expand(batch_size, -1) < lengths.unsqueeze(1)

    rand_mat = torch.rand(batch_size, length, device=device)
    rand_mat = torch.where(mask, rand_mat, -torch.inf)
    k_max = k.max().item()
    _, topk_indices = torch.topk(rand_mat, k_max, dim=1)

    scores = torch.arange(length,device=device,dtype=torch.float).expand(batch_size,-1).clone()

    shifts = torch.randint(-2, 3, (batch_size, k_max), device=device)
    original_pos = topk_indices
    

    move_mask = torch.arange(k_max, device=device).unsqueeze(0) < k.unsqueeze(1)

    batch_idx = torch.arange(batch_size, device=device)[:, None].expand(-1, k_max)[move_mask]
    orig_flat = original_pos[move_mask]
    

    shift_flat = shifts[move_mask]

    scores[batch_idx,orig_flat] = (orig_flat.float()+shift_flat)-0.1
    scores = torch.where(mask, scores, torch.tensor(float('inf'), device=device))
    sorted_idx = torch.argsort(scores, dim=1)
    sorted_idx_expanded = sorted_idx.unsqueeze(-1).expand(-1, -1, x.shape[2])
    new_x = torch.gather(x, 1, sorted_idx_expanded)

    

    return new_x


def pre_aug2(x,lengths,strength = 0.2,noise_scale=0.02):
    
    batch_size = x.shape[0]
    length = x.shape[1]
    device = x.device

    k = (lengths.float() * strength).floor().long()
    mask = (torch.arange(length, device=device).expand(batch_size, -1) < lengths.unsqueeze(1)).unsqueeze(-1)
    valid_x = x*mask.float()
    flow_mean = valid_x.sum(dim=1)/lengths.unsqueeze(-1).clamp(min=1)
    sq_diff = ((x-flow_mean.unsqueeze(1))**2)*mask.float()
    flow_var = sq_diff.sum(dim=1) / lengths.unsqueeze(-1).clamp(min=1)
    flow_std = torch.sqrt(flow_var) 

    rand_mat = torch.rand(batch_size, length, device=device)
    rand_mat = torch.where(mask.squeeze(-1), rand_mat, -torch.inf)
    k_max = k.max().item()
    _, topk_indices = torch.topk(rand_mat, k_max, dim=1)

    
    valid_mask = torch.arange(k_max, device=device).unsqueeze(0) < k.unsqueeze(1)

    batch_idx = torch.arange(batch_size, device=device)[:, None].expand(-1, k_max)[valid_mask]

    selected_std = flow_std[batch_idx]
    noise = torch.randn(len(batch_idx), x.shape[2], device=device) * selected_std * noise_scale

    pos_flat = topk_indices[valid_mask]
    

    new_x = x.clone()
    new_x[batch_idx, pos_flat] += noise

    new_x = torch.clamp(new_x, min=0.0)
    return new_x

def mixup_data(x, y, lengths, alpha=1.0,window=10,mode='ours'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    length = x.size()[1]
    index = torch.randperm(batch_size)
    device = x.device

    if mode == 'mixup':
        mixed_x = lam * x + (1 - lam) * x[index, :].clone().detach()
        y_a, y_b = y, y[index].clone().detach()
        return mixed_x, index, y_a, y_b, lam


    n_fft = window
    x_reshape = x.reshape(batch_size,length // n_fft,n_fft,x.shape[2])
    fft_result1 = torch.fft.fft(x_reshape)
    fft_result2 = torch.fft.fft(x_reshape[index])
    cutoff_frequency = np.random.beta(alpha,alpha)
    lam = cutoff_frequency
    
    cutoff_index = int(cutoff_frequency * (n_fft // 2))
    
    filtered_fft_result = fft_result1.clone()
    filtered_fft_result[:,:,cutoff_index:n_fft - cutoff_index+1,:] = 0
    rec_sig_low1 = torch.fft.ifft(filtered_fft_result)
    filtered_fft_result = fft_result1.clone()
    filtered_fft_result[:,:,:cutoff_index,:] = 0
    filtered_fft_result[:,:,n_fft - cutoff_index+1:,:] = 0
    rec_sig_high1 = torch.fft.ifft(filtered_fft_result)

    filtered_fft_result = fft_result2.clone()
    filtered_fft_result[:,:,cutoff_index:n_fft - cutoff_index+1,:] = 0
    rec_sig_low2 = torch.fft.ifft(filtered_fft_result)
    filtered_fft_result = fft_result2.clone()
    filtered_fft_result[:,:,:cutoff_index,:] = 0
    filtered_fft_result[:,:,n_fft - cutoff_index+1:,:] = 0
    rec_sig_high2 = torch.fft.ifft(filtered_fft_result)
    
    mixed_x = torch.real(rec_sig_low1 + rec_sig_high2)

    y_a, y_b = y, y[index].clone().detach()
    mixed_x = mixed_x.reshape(batch_size,length,x.shape[2])

    if mode!='freq_mix':        
        indices = torch.nonzero(y_a!=y_b).squeeze()
        mixed_x[indices] = lam * x[indices] + (1 - lam) * x[index, :][indices]
    
    mask = torch.arange(length,device=device).expand(batch_size,length) < lengths.unsqueeze(1)
    mixed_x[~mask] = 0

    mixed_x = mixed_x.clone().detach()
    
    
    return mixed_x, index, y_a, y_b, lam

def single_augmentation(x, y, mode='noise'):
    if mode == 'noise':
        device = x.device
        batch_size = x.shape[0]
        percentage = 0.05
        valid_mask = (x[:,:,0]!=0)
        lengths = valid_mask.sum(dim=1).to(device)
        mask = (torch.arange(x.shape[1], device=device).expand(batch_size, -1) < lengths.unsqueeze(1)).unsqueeze(-1)
        valid_x = x*mask.float()
        flow_mean = valid_x.sum(dim=1)/lengths.unsqueeze(-1).clamp(min=1)
        sq_diff = ((x-flow_mean.unsqueeze(1))**2)*mask.float()
        flow_var = sq_diff.sum(dim=1) / lengths.unsqueeze(-1).clamp(min=1)
        flow_std = torch.sqrt(flow_var)
        
        
        noise = torch.rand_like(x, device=device) * (flow_std.unsqueeze(1) * percentage)
        
        noisy_sequence = x + noise
        noisy_sequence = (noisy_sequence * mask).to(dtype=torch.float32)
        return noisy_sequence
    elif mode == 'random_mask':
        device = x.device
        B,T,F = x.shape
        valid_mask = (x[:,:,0]!=0)
        lengths = valid_mask.sum(dim=1).to(device)
        num_mask = (lengths * 0.2).long()
        rand = torch.rand(B,T,device=device)
        rand[~valid_mask] = float('inf')
        sorted_idx = rand.argsort(dim=1)
        mask = torch.zeros(B,T,device=device,dtype=torch.bool)
        row_idx = torch.arange(B,device=device).unsqueeze(1)
        k_max = num_mask.max()
        topk_idx = sorted_idx[:,:k_max]
        valid_k_mask = torch.arange(k_max, device=device).unsqueeze(0) < num_mask.unsqueeze(1)
        mask[row_idx, topk_idx] = valid_k_mask
        mask = mask.unsqueeze(-1).expand(-1, -1, F)
        x_masked = x.clone().detach()
        x_masked[mask] = 0
        return x_masked
    elif mode=='reperm':
        k = 5
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        device = x.device
        result = torch.zeros_like(x).to(device)
        lengths = (x[:,:,0]!=0).sum(dim=1).to(device)

        for i in range(batch_size):
            actual_length = lengths[i].item()
            segments = []
            segment_length = actual_length // k  
            extra_length = actual_length % k    

            
            start_idx = 0
            for j in range(k):
                end_idx = start_idx + segment_length + (extra_length if j == k - 1 else 0)  
                segments.append(x[i, start_idx:end_idx,:])
                start_idx = end_idx

            
            permuted_indices = torch.randperm(k)
            permuted_segments = [segments[j] for j in permuted_indices]

            
            rearranged_tensor = torch.cat(permuted_segments,dim=0)

            
            result[i, :len(rearranged_tensor),:] = rearranged_tensor
            result[i, len(rearranged_tensor):,:] = 0  
        return result

def train(model:nn.Module,class_model:nn.Module,processor:SAMProcessor, train_loader:DataLoader,pretrain_loader:DataLoader, val_loader:DataLoader,max_len,lr,savedir, restart_file=None, test_loader = None,preaug_ratio=0.2,args=None):
    
    
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch=0
    
    num_epochs = 5
    best_val_loss = float('inf')
    early_stop=0
    filemode = 'w'
    scheduler=None
    if restart_file is not None:
        filemode = 'a'
    if (restart_file is not None) and ('contrastive' in restart_file):
        model,optimizer,scheduler,start_epoch,loss,best_val_loss = load_checkpoint(savedir + '/'+restart_file,model,optimizer,scheduler)
        start_epoch += 1
        filemode = 'a'
    

    logging.basicConfig(
        filename=savedir + '/training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode=filemode
    )
    j = 0
    device = 'cuda'
    if not ((restart_file is not None) and ('classifier' in restart_file)):

        print('==================Contrastive Training Start=========================')
        logging.info('==================Contrastive Training Start=========================')
        
        for epoch in range(start_epoch,num_epochs):
            print('epoch:',epoch)
            model.train()
            start = time.perf_counter()
            for inputs, pos, targets in pretrain_loader:              
                
                x_i = processor.get_sam(inputs,only_pos=False,give_pos=True,position=pos).to(device)
                
                aug_inputs = traffic_augmentation(inputs)
            
                x_j = processor.get_sam(aug_inputs,only_pos=False,give_pos=False).to(device)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _,z_i = model(x_i)
                    _,z_j = model(x_j)
                    

                if MIXUP:
                    window = args.window
                    inputs = inputs.to(device)
                    sequences = inputs[:,[0,2],:]
                    length = (inputs[:,0,:]!=0).sum(dim=1).to(device)
                    sequences = sequences.permute(0,2,1)
                    if PRE_AUG:
                        sequences = pre_aug(sequences,length,strength=preaug_ratio)
                        sequences = pre_aug2(sequences,length,strength=preaug_ratio)
                    if args.aug == 'freq_mix':
                        window=100
                    sequences, indices, targets_a, targets_b, lam = mixup_data(sequences, targets,length,alpha=args.alpha,window=window,mode=args.aug)
                    sequences = sequences.permute(0,2,1)
                    sequences = torch.cat((sequences[:,0:1,:],inputs[:,1:2,:],sequences[:,1:2,:]),dim=1)
                    x_mix = processor.get_sam(sequences,only_pos=False,give_pos=False).to(device)
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        _,z_mix = model(x_mix)
                    z_k = z_i[indices]
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        loss = criterion(z_i,z_j) + lam * criterion(z_i,z_mix) + (1-lam) * criterion(z_k,z_mix)
                elif SINGLE_AUG:
                    mode = args.aug
                    inputs = inputs.to(device)
                    sequences = inputs[:,[0,2],:]
                    sequences = sequences.permute(0,2,1)
                    sequences = single_augmentation(sequences,targets,mode=mode)
                    sequences = sequences.permute(0,2,1)
                    sequences = torch.cat((sequences[:,0:1,:],inputs[:,1:2,:],sequences[:,1:2,:]),dim=1)
                    x_k = processor.get_sam(sequences,only_pos=False,give_pos=False).to(device)
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        _,z_k = model(x_k)
                        loss = criterion(z_i,z_j) + criterion(z_i,z_k)
                        
                else:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        loss = criterion(z_i,z_j)
                
                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            end = time.perf_counter()
            print(f'one epoch in {end - start}')
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}')
            logging.info(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}')
            if ((epoch - start_epoch)%20 == 0) or epoch == num_epochs-1:
                save_checkpoint(model,optimizer,None,epoch,loss,best_val_loss,savedir+f'/contrastive_checkpoint_epoch_{epoch}.pth')

    print('==================Classifier Training Start=========================')
    logging.info('==================Classifier Training Start=========================')
    classifier_model.freeze_encoder()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(class_model.parameters(), lr=lr)
    start_epoch=0
    
    num_epochs = 200
    best_val_loss = float('inf')
    early_stop=0
    filemode = 'w'
    scheduler=None
    if (restart_file is not None) and ('classifier' in restart_file):
        class_model,optimizer,scheduler,start_epoch,loss,best_val_loss = load_checkpoint(savedir + '/'+restart_file,class_model,optimizer,scheduler)
        start_epoch += 1

    
    print('Begin train loader cache...')
    start = time.perf_counter()
    train_loader.dataset.cache_features(class_model)
    end = time.perf_counter()
    print(f'Finish cache in {end-start}s.')

    print('Begin val loader cache...')
    start = time.perf_counter()
    val_loader.dataset.cache_features(class_model)
    end = time.perf_counter()
    print(f'Finish cache in {end-start}s.')

    class_model.only_classifier = True
    
    for epoch in range(start_epoch,num_epochs):
        print('epoch:',epoch)
        class_model.train()
        start = time.perf_counter()
        for inputs, pos, features, targets in train_loader:
        
            features = features.to(device)        
            targets = targets.to(device)
            optimizer.zero_grad()
            
            outputs = class_model(features)
                
            loss = criterion(outputs,targets)
            loss.backward()
           
            optimizer.step()
        end = time.perf_counter()
        
        print(f'one epoch in {end - start}')
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}')
        logging.info(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}')
        
        
        class_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, pos, features, targets in val_loader:            
                features = features.to(device)
                targets = targets.to(device)   
                outputs = class_model(features)
                loss = criterion(outputs, targets)

                val_loss += loss.item()  
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        print(
        f'Epoch [{epoch}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        logging.info(f'Epoch [{epoch}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        
        if (epoch - start_epoch)%30 == 0:
            save_checkpoint(class_model,optimizer,None,epoch,loss,best_val_loss,savedir+f'/classifier_checkpoint_epoch_{epoch}.pth')
        if val_loss < best_val_loss or best_val_loss == float('inf'):
            early_stop=0
            best_val_loss = val_loss
            torch.save(class_model.state_dict(), savedir + '/best.pth')
            print(f"Best model saved in epoch {epoch}.")
            logging.info(f"Best model saved in epoch {epoch}.")
        else:
            early_stop += 1
            if early_stop >= EARLY_STOP_EPOCH:
                print(f'Early stop in epoch {epoch}, because val loss does not change for {EARLY_STOP_EPOCH} epochs')
                logging.info(f'Early stop in epoch {epoch}, because val loss does not change for {EARLY_STOP_EPOCH} epochs')
                save_checkpoint(class_model,optimizer,None,epoch,loss,best_val_loss,savedir+f'/classifier_checkpoint_epoch_{epoch}.pth')
                break
    class_model.only_classifier = False


def test(model:nn.Module,processor:SAMProcessor,test_loader:DataLoader,max_len,savedir=None,target_file=None,description='ori test:'):
    device = 'cuda'
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    if target_file is not None:
        model,optimizer,_,start_epoch,loss,best_val_loss = load_checkpoint(savedir + '/'+target_file,model,optimizer,None)
    else:
        model.load_state_dict(torch.load(savedir + '/best.pth'))

    model.eval()

    logging.basicConfig(
        filename=savedir + '/training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )  
    all_targets = []
    all_predictions = []

    wrong = []
    with torch.no_grad():
        for inputs, pos, targets in test_loader:
            inputs = processor.get_sam(inputs,only_pos=False,give_pos=True,position=pos).to(device)      
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_targets.extend(targets.cpu().numpy())  
            all_predictions.extend(predicted.cpu().numpy())  

            if inputs[(targets!=predicted) & (targets==1)].size(0)!=0:
                wrong.append(inputs[(targets!=predicted) & (targets==1)])

    correct = sum(np.array(all_predictions) == np.array(all_targets))
    total = len(all_targets)
    accuracy = 100 * correct / total
    print(description)
    logging.info(description)
    if restart_file is not None:
        print(f'model:{restart_file}')
        logging.info(f'model:{restart_file}')
    print(f'Test Accuracy: {accuracy:.2f}%')
    logging.info(f'Test Accuracy: {accuracy:.2f}%')

    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    conf = confusion_matrix(all_targets, all_predictions)
    tn, fp, fn, tp = conf.ravel()
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'FPR: {fpr:.4f}')
    print(f'Confusion matrix:\n',conf)

    logging.info(f'Precision: {precision:.4f}')
    logging.info(f'Recall: {recall:.4f}')
    logging.info(f'F1 Score: {f1:.4f}')
    logging.info(f'FPR: {fpr:.4f}')
    logging.info(f'Confusion matrix:\n{conf}')
    logging.info(f'final_result={f1}')

    return accuracy, f1   

def read_dataframe(input_path):
    df = pd.read_csv(input_path, dtype=flow_column_types)
    df['dir_sequences'] = df['dir_sequences'].apply(lambda x: list(map(int, x.split('_'))))
    df['len_sequences'] = df['len_sequences'].apply(lambda x: list(map(int, x.split('_'))))
    df['time_interval_sequences'] = df['time_interval_sequences'].apply(lambda x: list(map(float, x.split('_'))))
    sequences = df['len_sequences'].values.tolist()
    iats = df['time_interval_sequences'].values.tolist()
    directions = df['dir_sequences'].values.tolist()
    labels = df['label'].to_numpy()
    return sequences,iats,directions,labels
    

def transform_to_tensor(sequences, iats, directions,pad_size):
    sequences = [
        (lst[:pad_size] + [0] * (pad_size - len(lst))) if len(lst) < pad_size
        else lst[:pad_size]
        for lst in sequences
    ]
    iats = [
        (lst[:pad_size] + [0] * (pad_size - len(lst))) if len(lst) < pad_size
        else lst[:pad_size]
        for lst in iats
    ]
    directions = [
        (lst[:pad_size] + [0] * (pad_size - len(lst))) if len(lst) < pad_size
        else lst[:pad_size]
        for lst in directions
    ]
    
    x_seq = torch.tensor(sequences, dtype=torch.float32)
    x_iat = torch.tensor(iats, dtype=torch.float32)
    x_dir = torch.tensor(directions, dtype=torch.float32)

    X = torch.stack([x_seq, x_dir, x_iat], dim=1)
    y = torch.tensor(labels,dtype=torch.long)
    
    return X, y


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--aug',type=str,default='noise')
    parser.add_argument('--train_path',type=str,default="train.csv")
    parser.add_argument('--test_path',type=str,default="test.csv")
    parser.add_argument('--save_dir',type=str,default="./info")
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--wv_path',type=str,default="./wv")
    parser.add_argument('--alpha',type=float,default=1.0)
    parser.add_argument('--window',type=int,default=10)

    K = 100

    args = parser.parse_args()
    assert args.aug in ['noaug','ours','freq_mix','mixup','noise','random_mask','reperm']

    if args.aug in ['ours','freq_mix','mixup']:
        MIXUP = True
    else:
        MIXUP = False

    if args.aug == 'ours':
        PRE_AUG = True
    else:
        PRE_AUG = False

    if args.aug in ['noise','random_mask','reperm']:
        SINGLE_AUG = True
    else:
        SINGLE_AUG = False

    random_seed = args.seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print('Seed:',random_seed)
    print(f'alpha:{args.alpha}')
    print(f'window:{args.window}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequences,iats,directions,labels = read_dataframe(args.train_path)
    X,y = transform_to_tensor(sequences,iats,directions,pad_size=K)

    print('Number:',X.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)   
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,test_size=0.5, random_state=42,stratify=y_test)

    idx_benign = (y_train == 0).nonzero(as_tuple=True)[0]
    idx_malicious = (y_train > 0).nonzero(as_tuple=True)[0]
    rand_idx_b = torch.randperm(len(idx_benign))[:8000]
    sample_idx_benign = idx_benign[rand_idx_b]
    rand_idx_m = torch.randperm(len(idx_malicious))[:2000]
    sample_idx_malicious = idx_malicious[rand_idx_m]
    final_indices = torch.cat([sample_idx_benign, sample_idx_malicious])
    X_sampled = X_train[final_indices]
    y_sampled = y_train[final_indices]
        
    processor = SAMProcessor(K=K,B=100,seed=random_seed)

    processor.train_embeddings(X_train.numpy(),model_dir=args.wv_path)
    
    train_dataset = SmartDetectorDataset(X_train, y_train,processor)
    val_dataset = SmartDetectorDataset(X_val, y_val,processor)
    test_dataset = SmartDetectorDataset(X_test, y_test,processor)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    sample_dataset = SmartDetectorDataset(X_sampled,y_sampled,processor)
    pretrain_loader = DataLoader(sample_dataset,batch_size=batch_size,shuffle=True)
    

    should_train = True
    restart_file = None

    if (not should_train) or (restart_file is not None):
        filemode = 'a'
    else:
        filemode = 'w'

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Folder {save_dir} create finished.")
    else:
        print(f"Folder {save_dir} already exists.")
    logging.basicConfig(
        filename=save_dir + '/training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode=filemode
    )
    logging.info(f'SEED:{random_seed}')
    logging.info(f'Dir:{args.save_dir}')
    logging.info(f'aug:{args.aug}')
    logging.info(f'alpha:{args.alpha}')
    logging.info(f'window:{args.window}')
    

    lr = 0.0001

    contrastive_model = SmartDetectorEncoder(embedding_dim=128).to(device)
    classifier_model = Classifier(num_classes=2,encoder=contrastive_model).to(device)

    if should_train:
        train(contrastive_model,classifier_model,processor,train_loader,pretrain_loader,val_loader,lr=lr,max_len=K,savedir=save_dir,restart_file=restart_file,args=args)
    
    test(classifier_model,processor,test_loader,max_len=K,savedir=save_dir,target_file=restart_file,description='ID test:')

    sequences,iats,directions,labels = read_dataframe(args.test_path)
    X,y = transform_to_tensor(sequences,iats,directions,pad_size=K)
    test_dataset = SmartDetectorDataset(X, y,processor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test(classifier_model,processor,test_loader,max_len=K,savedir=save_dir,target_file=restart_file,description='OOD test:')




