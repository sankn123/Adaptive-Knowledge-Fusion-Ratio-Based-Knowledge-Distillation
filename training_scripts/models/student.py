import torch
import torch.nn as nn
import sys
import timm

class Student(nn.Module): 
    def __init__(self, student_model_name, num_classes, pretrained):
        super(Student, self).__init__()
        self.student_model_name = student_model_name
        pretrained_model = timm.create_model(student_model_name, pretrained=pretrained) 
        
        Pre_Trained_Layers=nn.Sequential(*list(pretrained_model.children())[:-1])
        self.features=Pre_Trained_Layers
        if 'vit' in student_model_name:
            out = pretrained_model.head.in_features
        elif 'resnet' in student_model_name:
            out = pretrained_model.fc.in_features
        else:
            print(f"Student Model {student_model_name} not implemented")
            sys.exit()
        self.fc=nn.Linear(out,num_classes)  
        
    def forward(self,image):
        x1 = self.features(image)
        x2=self.fc(x1)
        if "vit" in self.student_model_name:
            x2 = torch.mean(x2,dim=1)
        
        return x2