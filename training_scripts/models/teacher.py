import torch
import torch.nn as nn
class Teacher(nn.Module): 
    def __init__(self,MLP_model, Num_Classes):
        super(Teacher, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(MLP_model.children())[:-1])
        
        self.features=Pre_Trained_Layers
        
        out = MLP_model.head.in_features
        
        self.fc=nn.Linear(out,Num_Classes)  
        
    def forward(self,image):
        x1 = self.features(image) 
        x2=self.fc(x1)
        #x2=x2.mean(dim=1)
        return x2