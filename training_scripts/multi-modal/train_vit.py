import numpy as np 
import torch 
import timm

import sys
sys.path.append("/export/home/vivian/sankn/AdaptiveKD/training_scripts")

from utils.custom_dataloaders import *
from utils.pytorchtools import EarlyStopping
from models.multimodal import basic_fusion

import random
import torch.optim as optim 
from torch.utils.data import dataset, DataLoader 
import torchvision.transforms as transforms 
import pandas as pd 
import torch.nn as nn 
import torch.nn.functional as F
import time # import time 
import sys # Import System 
import os # Import OS
import warnings
warnings.filterwarnings("ignore")
import sklearn

import os 
from sklearn.metrics import precision_recall_fscore_support
import logging


dataset = "advance"

teacher_model_name = "vit_base_patch16_224"
# teacher_model_name = "resnet18"

match dataset:
           
    case "advance":
        from utils.paths.advance_pathname import *
        Num_Classes = 13
        r = 0.9 # train/valid split
        learning_rate=2e-5
        batch_size=16
        
    case _:
        print(f"{dataset} Not implemented")
        sys.exit()


log_path = os.path.join(save_path,f'model_{teacher_model_name}_bs{batch_size}_lr{learning_rate}')
os.makedirs(log_path,exist_ok = True)

MODEL_SAVE_PATH= os.path.join(log_path, "multimodal_model.pt")


logfile = os.path.join(log_path, "output.log")




logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(message)s",
handlers=[
        logging.FileHandler(logfile, "w"),
        logging.StreamHandler()
])

SEED = 1234 # Initialize seed 
EPOCHS=1000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda') # Define device type 
# warnings.filterwarnings("ignore")
data_transformations = transforms.Compose([ # Training Transform 
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])])

train_dataset=Train_Data(Train_Label_Path, Train_Data_Path, H5py_Path, transform=data_transformations, is_multimodal=True) 

train_size = int(r * len(train_dataset)) 
valid_size = len(train_dataset) - train_size 
Train_Dataset,Valid_Dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size]) 

Test_Dataset=Test_Data(Test_Label_Path, Test_Data_Path, H5py_Path,transform=data_transformations, is_multimodal=True)

train_loader=DataLoader(dataset=Train_Dataset,batch_size=batch_size,shuffle=True) # Create Training Dataloader 
valid_loader=DataLoader(dataset=Valid_Dataset,batch_size=batch_size,shuffle=False)# Create Validation Dataloader 
test_loader=DataLoader(dataset=Test_Dataset,batch_size=batch_size,shuffle=False) # Create Test Dataloader 


MLP_model = timm.create_model(teacher_model_name, pretrained=True)

# print(MLP_model)
# sys.exit()
Teacher_Model = basic_fusion(MLP_model, Num_Classes, teacher_model_name)

Teacher_Model=Teacher_Model.to(device)
Teacher_optimizer = optim.Adam(Teacher_Model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Define Loss Function 

def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

def train(model,device,iterator, optimizer, criterion): 
    # early_stopping = EarlyStopping(patience=7, verbose=True)
    
    epoch_loss = 0
    epoch_acc = 0
    count=0
    model.train() # call model object for training 
    for (aud,img,y) in iterator:
        aud=aud.float()
        aud=torch.cat([aud,aud,aud],dim=1)
        
        img = img.float()
        
        aud=aud.to(device)
        img = img.to(device)        
     
        y = y.type(torch.LongTensor)
        y=y.to(device)# Transfer label  to device

        optimizer.zero_grad() # Initialize gredients as zeros 
        count=count+1
        #print(x.shape)
        Predicted_Train_Label=model(aud,img)
        # print(Predicted_Train_Label.shape)
        Predicted_Train_Label = Predicted_Train_Label.mean(dim=1)

        # print(Predicted_Train_Label.shape, y.shape)
        loss = criterion(Predicted_Train_Label, y) # training loss
        acc = calculate_accuracy(Predicted_Train_Label, y) # training accuracy
        loss.backward() # backpropogation 
        optimizer.step() # optimize the model weights using an optimizer 
        epoch_loss += loss.item() # sum of training loss
        epoch_acc += acc.item() # sum of training accuracy  
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model,device,iterator, criterion): # Evaluate Validation accuracy 
    #print("Validation Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    precision=0
    recall=0
    fscore=0
    model.eval() # call model object for evaluation 
    
    with torch.no_grad(): # Without computation of gredient 
        for (aud,img,y) in iterator:
            aud=aud.float()
            aud=torch.cat([aud,aud,aud],dim=1)
            
            img = img.float()
            
            aud=aud.to(device)
            img = img.to(device)
            y = y.type(torch.LongTensor)
            y=y.to(device)# Transfer label  to device
            count=count+1
            Predicted_Label = model(aud,img) # Predict claa label 
            Predicted_Label=Predicted_Label.mean(dim=1)
            
            loss = criterion(Predicted_Label, y) # Compute Loss 
            acc = calculate_accuracy(Predicted_Label, y) # compute Accuracy
            Predicted_Label_2=Predicted_Label.detach().cpu().numpy()
            Predicted_Label_2=np.argmax(Predicted_Label_2,axis=1)
            y_2=y.detach().cpu().numpy()
            precision1, recall1, fscore1, sup = sklearn.metrics.precision_recall_fscore_support(y_2, Predicted_Label_2, average='weighted')

            epoch_loss += loss.item() # Compute Sum of  Loss 
            epoch_acc += acc.item() # Compute  Sum of Accuracy
            precision=precision + precision1
            recall=recall+recall1
            fscore=fscore+fscore1

        
    return epoch_loss / len(iterator), epoch_acc / len(iterator) , precision/len(iterator), recall/len(iterator), fscore/len(iterator)
 
best_valid_loss = float('inf')

logging.info("Training ...") 
total_time=0
logging.info("---------------------------------------------------------------------------------------------------------------------")   
early_stopping = EarlyStopping(patience=7, verbose=True) # early Stopping Criteria
for epoch in range(EPOCHS):
    start_time=time.time() # Compute Start Time 
    train_loss, train_acc = train(Teacher_Model,device,train_loader,Teacher_optimizer, criterion) # Call Training Process 
    train_loss=round(train_loss,2) # Round training loss 
    train_acc=round(train_acc,2) # Round training accuracy 
    valid_loss, valid_acc,_,_,_ = evaluate(Teacher_Model,device,valid_loader,criterion) # Call Validation Process 
    valid_loss=round(valid_loss,2) # Round validation loss
    valid_acc=round(valid_acc,2) # Round accuracy 
    end_time=(time.time()-start_time) # Compute End time 
    end_time=round(end_time,2)  # Round End Time 
    total_time=total_time+end_time
    logging.info(f" | Epoch={epoch} | Training Accuracy={train_acc*100} | Validation Accuracy= {valid_acc*100} | Training Loss= {train_loss} | Validation_Loss= {valid_loss} Time Taken(Seconds)={end_time}|")
    logging.info("---------------------------------------------------------------------------------------------------------------------")
    
    early_stopping(valid_loss,Teacher_Model,MODEL_SAVE_PATH) # call Early Stopping to Prevent Overfitting 
    if early_stopping.early_stop:
        logging.info("Early stopping")
        logging.info(f'Total Time: {total_time} sec')
        break
    Teacher_Model.load_state_dict(torch.load(MODEL_SAVE_PATH))

Teacher_Model.load_state_dict(torch.load(MODEL_SAVE_PATH)) # load the trained model 
test_loss, test_acc,p,r,f1 = evaluate(Teacher_Model, device, test_loader, criterion) # Compute Test Accuracy on Unseen Signals 
test_loss=round(test_loss,2)# Round test loss
test_acc=round(test_acc,2) # Round test accuracy

p=round(p,3) 
r=round(r,3) 
f1=round(f1,3)

logging.info(f"|Test Loss= {test_loss} Test Accuracy= {test_acc*100}") # print test accuracy 
logging.info(f"P: {p}, R: {r}, F1: {f1}")  

