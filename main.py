import os.path
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time
import argparse
import logging
from TimeseriesData import TimeSeriesDataset
from models.Transformer import Model 
from models.TMWF import TMWF_DFNet
from models.BAPM import BAPM
from models.CNN import CNN1d
from models.DF import DeepFinger
import yaml
from Rosseta.resnet_base_network import ResNet18
import torch.nn.functional as F
import os

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

EARLY_STOP_EPOCH = 20
USE_TRANSFORMER = False
MIX = False
PRE_AUG = False
SINGLE_AUG = False
ROSETTA = False

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x,lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        out, (ht,ct) = self.lstm(packed_input)
        out = self.fc(ht[-1])
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

def single_augmentation(x, y , mode='noise'):
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

def generate_encoder():
    device = 'cuda'
    config = yaml.load(open("../Rosseta/model/checkpoints/config.yaml", "r"), Loader=yaml.FullLoader)
    encoder = ResNet18(**config['network'])

    load_params = torch.load(os.path.join('../Rosseta/model/checkpoints/model.pth'),
                            map_location=torch.device(torch.device(device)))

    if 'online_network_state_dict' in load_params:
        encoder.load_state_dict(load_params['online_network_state_dict'])
        print("Parameters successfully loaded.")
    encoder = torch.nn.Sequential(*list(encoder.children())[:-1])    
    encoder = encoder.to(device)
    encoder.eval()
    return encoder

def train(model:nn.Module, train_loader:DataLoader, val_loader:DataLoader,max_len,lr,savedir,args=None):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_epoch=0
    
    num_epochs = 200
    best_val_loss = float('inf')
    early_stop=0
    filemode = 'a'

    logging.basicConfig(
        filename=save_dir + '/training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode=filemode
    )

    if ROSETTA:
        encoder = generate_encoder()
        proj = nn.Linear(512,max_len).to('cuda')
        for param in encoder.parameters():
            param.requires_grad = False

    for epoch in range(start_epoch,num_epochs):
        model.train()
        start = time.perf_counter()
        for inputs, targets,length in train_loader:

            if ROSETTA:
                batch_size = inputs.shape[0]
                seq_len = max_len
                target_len = 300
                pad_len = target_len - seq_len
                if pad_len>0:
                    inputs = F.pad(inputs,(0,pad_len),'constant',0)
                inputs = inputs.reshape(-1,3,10,10).to(device)
                inputs = encoder(inputs).squeeze().unsqueeze(1)
                
                inputs = proj(inputs)     
                

            if PRE_AUG or MIX or SINGLE_AUG:
                if inputs.dim() == 2:
                        inputs = inputs.unsqueeze(2)
            if PRE_AUG:
                inputs = pre_aug(inputs,length,strength=args.preaug_ratio)
                inputs = pre_aug2(inputs,length,strength=args.preaug_ratio)

            if MIX:
                window=args.window
                if args.aug == 'freq_mix':
                    window=100
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,length,alpha=args.alpha,window=window,mode=args.aug)
                targets_a = targets_a.view(-1).to(device)
                targets_b = targets_b.view(-1).to(device)
            elif SINGLE_AUG:
                inputs = single_augmentation(inputs,targets,mode=args.aug)
            
            if PRE_AUG or MIX or SINGLE_AUG:
                inputs = inputs.squeeze()
            
            inputs = inputs.view(-1,max_len)
            targets = targets.view(-1).to(device)
            length = length.view(-1)
            inputs = inputs.unsqueeze(-1).to(device) 

            

            optimizer.zero_grad()
    
            
            if USE_TRANSFORMER:
                batch_size,seq_len,_ = inputs.size()
                padding_masks = torch.arange(max_len).expand(batch_size, max_len) < length.unsqueeze(1)
                padding_masks = padding_masks.to(device)
                outputs = model(inputs,padding_masks,None,None)
                if MIX:
                    loss = lam*criterion(outputs,targets_a)+(1-lam)*criterion(outputs,targets_b)
                else:
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs,length)
                if MIX:
                    loss = lam*criterion(outputs,targets_a)+(1-lam)*criterion(outputs,targets_b)
                else:
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
            for inputs, targets, length in val_loader:
                if ROSETTA:
                    batch_size = inputs.shape[0]
                    seq_len = max_len
                    target_len = 300
                    pad_len = target_len - seq_len
                    if pad_len>0:
                        inputs = F.pad(inputs,(0,pad_len),'constant',0)
                    inputs = inputs.reshape(-1,3,10,10).to(device)
                    inputs = encoder(inputs).squeeze().unsqueeze(1)
                    inputs = proj(inputs)  
                inputs = inputs.view(-1, max_len)
                targets = targets.view(-1).to(device)
                length = length.view(-1)
                inputs = inputs.unsqueeze(-1).to(device)
                                  

                if USE_TRANSFORMER:
                    batch_size,seq_len,_ = inputs.size()
                    padding_masks = torch.arange(max_len).expand(batch_size, max_len) < length.unsqueeze(1)
                    padding_masks = padding_masks.to(device)
                    outputs = model(inputs, padding_masks,None,None)
                    loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs,length)
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
        
        if val_loss < best_val_loss:
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
                save_checkpoint(model,optimizer,epoch,loss,best_val_loss,savedir+f'/checkpoint_epoch_{epoch}.pth')
                break
        if (epoch - start_epoch) % 20 == 0:
            save_checkpoint(model,optimizer,epoch,loss,best_val_loss,savedir+f'/checkpoint_epoch_{epoch}.pth')

def test(model:nn.Module,test_loader:DataLoader,max_len,savedir=None,target_file=None,description='ori test:'):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    if target_file is not None:
        model,optimizer,epoch,loss,best_val_loss = load_checkpoint(savedir + '/'+target_file,model,optimizer)
    else:
        model.load_state_dict(torch.load(savedir + '/best.pth'))

    model.eval()

    logging.basicConfig(
        filename=save_dir + '/training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    if ROSETTA:
        encoder = generate_encoder()
        proj = nn.Linear(512,max_len).to('cuda')
        for param in encoder.parameters():
            param.requires_grad = False

    all_targets = []
    all_predictions = []

    wrong = []

    with torch.no_grad():
        for inputs, targets,length in test_loader:
            
            if ROSETTA:
                batch_size = inputs.shape[0]
                seq_len = max_len
                target_len = 300
                pad_len = target_len - seq_len
                if pad_len>0:
                    inputs = F.pad(inputs,(0,pad_len),'constant',0)
                inputs = inputs.reshape(-1,3,10,10).to(device)
                inputs = encoder(inputs).squeeze().unsqueeze(1)
                inputs = proj(inputs)
            inputs = inputs.view(-1, max_len)
            targets = targets.view(-1).to(device)
            length = length.view(-1)
            inputs = inputs.unsqueeze(-1).to(device)
               
                
                
            if USE_TRANSFORMER:
                batch_size,seq_len,_ = inputs.size()
                padding_masks = torch.arange(max_len).expand(batch_size, max_len) < length.unsqueeze(1)
                padding_masks = padding_masks.to(device)
                outputs = model(inputs,padding_masks,None,None)
                _, predicted = torch.max(outputs.data, 1)
            else:
                outputs = model(inputs,length)
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


def save_checkpoint(model,optimizer,epoch,loss,best_val_loss,filepath):
    checkpoint = {
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'loss':loss,
        'best_val_loss':best_val_loss,
        'random_state': random.getstate(), 
        'torch_random_state': torch.get_rng_state(),  
    }
    torch.save(checkpoint,filepath)

def load_checkpoint(filepath,model,optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    best_val_loss = checkpoint['best_val_loss']
    random.setstate(checkpoint['random_state'])
    torch.set_rng_state(checkpoint['torch_random_state'])
    return model, optimizer, epoch, loss,best_val_loss

def read_dataframe(input_path):
    df = pd.read_csv(input_path, dtype=flow_column_types)
    
    df['len_sequences'] = df['len_sequences'].apply(lambda x: list(map(int, x.split('_'))))
    
    df.reset_index(inplace=True, drop=True)
    sequences = df['len_sequences'].values.tolist()
    labels = df['label'].values.tolist()
    return df,sequences,labels

def create_model(model_name,max_len):
    if model_name == 'DF':
        model = DeepFinger(in_dim=max_len,num_class=2)
        return model
    elif model_name=='Transformer':
        class Configs:
            def __init__(self):
                self.task_name='classification'
                self.pred_len=96
                self.label_len=48
                self.enc_in=1
                self.dec_in=1
                self.d_model=64
                self.d_ff=256
                self.embed='timeF'
                self.freq='h'
                self.dropout=0.1
                self.factor=1
                self.n_heads = 8
                self.activation = 'gelu'
                self.e_layers = 3
                self.distil='store_false'
                self.c_out=7
                self.seq_len = max_len
                self.num_class=2
                self.d_layers=1
                self.top_k=1
                self.num_kernels=6
        configs = Configs()
        model = Model(configs)
        return model
    elif model_name=='CNN':
        model = CNN1d(in_dim=1,n_class=2)
        return model
    elif model_name=='lstm':
        input_size = 1  
        hidden_size = 64
        output_size = 2 
        num_layers = 2
        model = LSTMClassifier(input_size, hidden_size, output_size, num_layers)
        return model
    elif model_name=='TMWF_DFNet':
        model = TMWF_DFNet(embed_dim=256,nhead=4,dim_feedforward=256*4,num_encoder_layers=1,num_decoder_layers=1,
                           max_len=max_len,num_queries=6,cls=2,dropout=0.1)
        return model
    
    elif model_name=='BAPM':
        model = BAPM(num_classes=2,num_tab=1)
        return model



if __name__=='__main__':
    random_seed=42
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',type=str,
                        default=f'train.csv')
    parser.add_argument('--test_path',type=str,
                        default=f'test.csv')
    parser.add_argument('--model_name',type=str,
                        default='lstm') 
    parser.add_argument('--save_dir',type=str,                        
                        default=f'./info')
    parser.add_argument('--aug',type=str,default='no')
    parser.add_argument('--should_train',action='store_true')

    parser.add_argument('--alpha',type=float,default=1.0)
    parser.add_argument('--preaug_ratio',type=float,default=0.2)
    parser.add_argument('--window',type=int,default=10)
    
    parser.add_argument('--valid_size',type=float,default=0.2)
    parser.add_argument('--test_size',type=float,default=0.2)

    args = parser.parse_args()

    valid_size = args.valid_size
    test_size = args.test_size

    if args.aug in ['ours','freq_mix','mixup']:
        MIX = True
        if args.aug == 'ours':
            PRE_AUG = True
    elif args.aug == 'no':
        MIX = False
        PRE_AUG = False
    elif args.aug in ['noise','random_mask','reperm']:
        SINGLE_AUG = True
    elif args.aug in ['rosetta']:
        ROSETTA = True
    else:
        MIX = False
        PRE_AUG = False
        print('Unknown arg: aug. Use no augmentation.')

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_path = args.train_path
    max_len = 100

    df, X, y = read_dataframe(input_path)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Folder {save_dir} create finished.")
    else:
        print(f"Folder {save_dir} already exists.")
    

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=valid_size+test_size, random_state=42, stratify=y)    
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,test_size=test_size/(valid_size+test_size), random_state=42,stratify=y_test)
    
    
    train_dataset = TimeSeriesDataset(X_train, y_train,max_len=max_len)
    val_dataset = TimeSeriesDataset(X_val, y_val, max_len=max_len)
    test_dataset = TimeSeriesDataset(X_test, y_test,max_len=max_len)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=6) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    lr = 0.0001
    model = create_model(model_name=args.model_name,max_len=max_len)
    model = model.to(device)
    if args.model_name == 'Transformer':
        USE_TRANSFORMER = True
    

    logging.basicConfig(
        filename=save_dir + '/training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )

    logging.info(f'alpha:{args.alpha}')
    logging.info(f'preaug_ratio:{args.preaug_ratio}')
    logging.info(f'window:{args.window}')
    logging.info(f'valid_size:{valid_size}')
    logging.info(f'test_size:{test_size}')

    if args.should_train:
        train(model,train_loader,val_loader,lr=lr,max_len=max_len,savedir=save_dir,args=args)
    
    test(model,test_loader,max_len=max_len,target_file=None,savedir=save_dir,description='ID test:')

    
    df, X, y = read_dataframe(args.test_path)
    test_dataset = TimeSeriesDataset(X, y, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test(model,test_loader,max_len=max_len,savedir=save_dir,description='OOD test:')




