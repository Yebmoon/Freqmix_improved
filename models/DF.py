import torch.nn as nn

class DeepFinger(nn.Module):
    def __init__(self, in_dim, num_class):
        super(DeepFinger, self).__init__()

        self.conv1 = nn.Sequential(       
            nn.Conv1d(1, 32, 5, 1, 2),
            nn.BatchNorm1d(32), 
            nn.ELU(),                     
            nn.Conv1d(32, 32, 5, 1, 2),
            nn.BatchNorm1d(32), 
            nn.ELU(),           
            nn.MaxPool1d(3, 3, 0),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(       
            nn.Conv1d(32, 64, 5, 1, 2),
            nn.BatchNorm1d(64), 
            nn.ReLU(),                     
            nn.Conv1d(64, 64, 5, 1, 2),
            nn.BatchNorm1d(64), 
            nn.ReLU(),           
            nn.MaxPool1d(3, 3, 0),
            nn.Dropout(0.1)
        )
        self.out1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(704, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.out2 = nn.Sequential(
            nn.Linear(256, num_class)
        )

    def forward(self, x, length):
        x = x.squeeze(-1)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        output = self.out1(x)
        output = self.out2(output)
        return output