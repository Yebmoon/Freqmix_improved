import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
import pickle
from sklearn.model_selection import train_test_split
from MATECDataset import MATECDataset
from torch.utils.data import DataLoader
import random
import logging
import time
import pandas as pd
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
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
MIXUP = False
EARLY_STOP_EPOCH = 20
PRE_AUG = False
SINGLE_AUG = False
class Config(object):

    def __init__(self,dataset):
        self.model_name = 'MATEC'
        self.train_path = dataset + '/data/train.csv'                                
        self.dev_path = dataset + '/data/dev.csv'                                   
        self.test_path = dataset + '/data/test.csv'                                  
        self.class_list = [0,1]                                          
        self.save_path = 'MATEC/'+ self.model_name + '.ckpt'       
        self.log_path = 'MATEC/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

        self.dropout = 0.1                                             
        self.require_improvement = 10000                               
        self.num_classes = len(self.class_list)                        
        self.n_vocab = 0                                               
        self.num_epochs = 50                                         
        self.batch_size = 128                                          
        self.pad_size = 100                                              
        self.learning_rate = 0.0005                                      
        self.embed =  432          
        self.dim_model = 432
        self.num_head = 3
        self.num_encoder = 2


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
       

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5  
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        return out

class Add_Norm(nn.Module):
    def __init__(self, dim_model):
        super(Add_Norm, self).__init__()
        self.fc1 = nn.Linear(dim_model, dim_model)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, origin_x, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = out + origin_x  
        out = self.dropout(out)
        out = self.layer_norm(out)
        return out 
    

    
class Feed_Forward(nn.Module):
    def __init__(self, dim_model):
        super(Feed_Forward, self).__init__()
        self.conv = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv(x)
        return out
    
class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Feed_Forward(dim_model)
        self.add_norm1 = Add_Norm(dim_model)
        self.add_norm2 = Add_Norm(dim_model)
        
    def forward(self, x):
        
        out = self.attention(x)
        out = self.add_norm1(x,out)
        out_ = self.feed_forward(out)
        out = self.add_norm2(out,out_)
        return out


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.fc1 = nn.Linear(config.pad_size,config.dim_model)
        self.encoder1 = Encoder(config.dim_model, config.num_head, config.dropout)
        self.encoder2 = Encoder(config.dim_model, config.num_head, config.dropout)
        
        self.fc2 = nn.Linear(1296, config.num_classes)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        
        out = self.fc1(x)
        out = self.encoder1(out)
        out = self.encoder2(out)
        out = torch.flatten(out,start_dim=1)
        out = self.dropout(out)
        out = self.fc2(out)
    
        return out

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

    packet_mask = (x!=0).any(dim=-1)

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
        return mixed_x, y_a, y_b, lam


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
    
    return mixed_x, y_a, y_b, lam

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
    
    
def transform_to_tensor(sequences, iats, directions):
    sequences = [
        (lst[:config.pad_size] + [0] * (config.pad_size - len(lst))) if len(lst) < config.pad_size
        else lst[:config.pad_size]
        for lst in sequences
    ]
    iats = [
        (lst[:config.pad_size] + [0] * (config.pad_size - len(lst))) if len(lst) < config.pad_size
        else lst[:config.pad_size]
        for lst in iats
    ]
    directions = [
        (lst[:config.pad_size] + [0] * (config.pad_size - len(lst))) if len(lst) < config.pad_size
        else lst[:config.pad_size]
        for lst in directions
    ]
    
    x_seq = torch.tensor(sequences, dtype=torch.float32)
    x_iat = torch.tensor(iats, dtype=torch.float32)
    x_dir = torch.tensor(directions, dtype=torch.float32)

    
    X = torch.stack([x_seq, x_iat, x_dir], dim=1)
    y = torch.tensor(labels,dtype=torch.long)
    return X, y

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

def train(model:nn.Module, train_loader:DataLoader, val_loader:DataLoader,max_len,lr,savedir, restart_file=None, test_loader = None,preaug_ratio=0.2,**kwargs):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch=0
    
    num_epochs = 200
    best_val_loss = float('inf')
    early_stop=0
    filemode = 'w'
    if restart_file is not None:
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

    for epoch in range(start_epoch,num_epochs):
        print('epoch:',epoch)
        model.train()
        start = time.perf_counter()
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if MIXUP:
                size_sequences = inputs[:,0,:]
                length = (size_sequences!=0).sum(dim=1).to(device)   
                sequences = inputs[:,[0,1],:]
                
                sequences = sequences.permute(0,2,1)
                if PRE_AUG:
                    sequences = pre_aug(sequences,length,strength=preaug_ratio)
                    sequences = pre_aug2(sequences,length,strength=preaug_ratio)                
                mode = kwargs['args'].aug
                window = kwargs['args'].window
                alpha = kwargs['args'].alpha
                if mode == 'freq_mix':
                    window=100
                sequences, targets_a, targets_b, lam = mixup_data(sequences, targets,length,window=window,alpha=alpha,mode=mode)
                sequences = sequences.permute(0,2,1)
                inputs[:,[0,1],:] = sequences
                
                
                
                targets_a = targets_a.view(-1).to(device)
                targets_b = targets_b.view(-1).to(device)
            elif SINGLE_AUG:
                mode = kwargs['args'].aug
                sequences = inputs[:,[0,1],:]
                sequences = sequences.permute(0,2,1)
                sequences = single_augmentation(sequences,targets,mode=mode)
                inputs[:,[0,1],:] = sequences.permute(0,2,1)
            optimizer.zero_grad()
            if MIXUP:             
                outputs = model(inputs)
                loss = lam*criterion(outputs,targets_a)+(1-lam)*criterion(outputs,targets_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            loss.backward() 
            optimizer.step()
        end = time.perf_counter()
        print(f'one epoch in {end - start}')
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}')
        logging.info(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}')

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)    
                outputs = model(inputs)
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
        if (epoch-start_epoch)%20==0:
            save_checkpoint(model,optimizer,None,epoch,loss,best_val_loss,savedir+f'/checkpoint_epoch_{epoch}.pth')
        if val_loss < best_val_loss or best_val_loss == float('inf'):
            early_stop=0
            best_val_loss = val_loss
            torch.save(model.state_dict(), savedir + '/best.pth')
            print(f"Best model saved in epoch {epoch}.")
            logging.info(f"Best model saved in epoch {epoch}.")
        else:
            early_stop += 1
            if early_stop >= EARLY_STOP_EPOCH:
                print(f'Early stop in epoch {epoch}, because val loss does not change for {EARLY_STOP_EPOCH} epochs')
                logging.info(f'Early stop in epoch {epoch}, because val loss does not change for {EARLY_STOP_EPOCH} epochs')
                break
        

def test(model:nn.Module,test_loader:DataLoader,max_len,savedir=None,target_file=None,description='ori test:'):
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
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug',type=str,default='noaug')
    parser.add_argument('--train_path',type=str,default="train.csv")
    parser.add_argument('--test_path',type=str,default="test.csv")
    parser.add_argument('--save_dir',type=str,default="./info")
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--alpha',type=float,default=1.0)
    parser.add_argument('--window',type=int,default=10)

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
    print('SEED:',random_seed)
    print(f'alpha:{args.alpha}')
    print(f'window:{args.window}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config("/DoH")
    matec = Model(config).to(device)
    sequences,iats,directions,labels = read_dataframe(args.train_path)
    X, y = transform_to_tensor(sequences,iats,directions)
    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.4, random_state=42, stratify=y)   
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,test_size=0.5, random_state=42,stratify=y_test)
    train_dataset = MATECDataset(X_train, y_train,)
    val_dataset = MATECDataset(X_val, y_val)
    test_dataset = MATECDataset(X_test, y_test)
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    should_train = True
    restart_file=None
    if (restart_file is not None) or (not should_train):
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
    logging.info(f'seed:{random_seed}')
    logging.info(f'alpha:{args.alpha}')
    logging.info(f'window:{args.window}')

    lr = 0.0001
    max_len = 100
    
    if should_train:
        train(matec,train_loader,val_loader,lr=lr,max_len=max_len,savedir=save_dir,restart_file=restart_file,args=args)
    test(matec,test_loader,max_len=max_len,savedir=save_dir,target_file=restart_file,description='ID test:')

    sequences,iats,directions,labels = read_dataframe(args.test_path)
    X, y = transform_to_tensor(sequences,iats,directions)
    test_dataset = MATECDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test(matec,test_loader,max_len=max_len,savedir=save_dir,target_file=restart_file,description='OOD test:')


    


    
