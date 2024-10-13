import torch
import torch.nn as nn



class basic_fusion(nn.Module): 
    def __init__(self, MLP_model, Num_Classes, model_name):
        super(basic_fusion, self).__init__()
        
        self.features = nn.Sequential(*list(MLP_model.children())[:-1]) 
           
        if "resnet" in model_name:
            out = MLP_model.fc.in_features
        elif "vit" in model_name:
            out = MLP_model.head.in_features
            
            
        self.fc=nn.Linear(out,Num_Classes)  
        
    def forward(self,aud,img):
        # print(f"Img,aud shape: {img.shape}, {aud.shape}")
        aud_feat =  self.features(aud)
        # print(f"aud feat shape: {aud_feat.shape}")
        img_feat =  self.features(img)
        # print(f"img feat shape: {img_feat.shape}")
        
        x = aud_feat+img_feat

        
        x = self.fc(x) 
        # print(f"returning: {x.shape}")
        return x

# class basic_fusion(nn.Module): 
#     def __init__(self,MLP_model_aud,  MLP_model_img, Num_Classes):
#         super(basic_fusion, self).__init__()
        
#         self.features_aud = nn.Sequential(*list(MLP_model_aud.children())[:-1])
        
#         self.features_img = nn.Sequential(*list(MLP_model_img.children())[:-1])
        

        
#         out = MLP_model_aud.head.in_features
        
#         self.fc=nn.Linear(out,Num_Classes)  
        
#     def forward(self,aud,img):
#         aud_feat =  self.features_aud(aud)
#         img_feat =  self.features_img(img)
        
#         x = aud_feat+img_feat
        
#         x = self.fc(x) 
#         return x