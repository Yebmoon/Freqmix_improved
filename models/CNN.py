import torch.nn as nn

class CNN1d(nn.Module):
    def __init__(self, in_dim, n_class):
        super(CNN1d, self).__init__()

        self.conv1 = nn.Sequential(       
            nn.Conv1d(in_channels=in_dim, out_channels=32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32), 
            nn.ReLU(),                     
            nn.MaxPool1d(kernel_size=8, stride=4, padding=0),
            nn.Dropout(0.2), 
        )

        self.fc = nn.Linear(22*32,n_class)           

    def forward(self, x, length):
        x = x.squeeze(-1)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x