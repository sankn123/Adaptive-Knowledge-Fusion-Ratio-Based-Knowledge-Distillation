'''
Basic ANN model for training alpha
'''
import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self,batch_size, num_classes):
        super().__init__()
        self.batch_size=batch_size
        self.num_classes=num_classes
        self.ff1=nn.Linear(batch_size*6*num_classes,512)

        self.ff2=nn.Linear(512,256)
        self.ff3=nn.Linear(256,128)
        self.ff4=nn.Linear(128,1)

        self.dropout=nn.Dropout(0.4)
        
    
    def forward(self,x):
        x=self.ff1(x)
        x=self.dropout(x)
        x=self.ff2(x)
        x=self.dropout(x)
        x=self.ff3(x)
        x=self.dropout(x)
        x=self.ff4(x)

        return torch.sigmoid(x)