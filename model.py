import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(FeatureExtractor, self).__init__()
        if bidirectional:
           self.hsize= 2
        else:
           self.hsize=1
        self.hidden_size = hidden_size
        self.num_layers= num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True,bidirectional=bidirectional)
       
        
    def forward(self, x):
       
        h0 = torch.zeros(self.hsize*self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.hsize*self.num_layers, x.size(0), self.hidden_size).to(x.device)

        _, (hn,_) = self.lstm(x, (h0, c0))
        hn= hn.transpose(1,0) 
        hn = hn.reshape(hn.shape[0],-1)
       #(hsize*num_layer,bs,hidden_size)
        return hn
    

class DomainClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        #x = self.sigmoid(x)
        return x

class LabelPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LabelPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


    
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


