import numpy as np 
import torch 
import timm

import sys
sys.path.append("<path to training_scripts folder")
from utils.custom_dataloaders import *
from utils.pytorchtools import EarlyStopping
from models.multimodal import basic_fusion
from models.ANN import ANN

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

import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.ticker import MaxNLocator

import os 
from sklearn.metrics import precision_recall_fscore_support
import logging


dataset = "advance"

is_student_pretrained = True

teacher_model_name = "<teacher_model_name>"
student_model_name = "<student_model_name>"

teacher_weights_path = "<teacher model weights>"

match dataset:
           
    case "advance":
        from utils.paths.advance_pathname import *
        Num_Classes = 13
        r = 0.9 # train/valid split
        learning_rate=5e-4
        batch_size=120
        
    case _:
        print(f"{dataset} Not implemented")
        sys.exit()



Temprature=200



log_path = os.path.join(save_path,"AKD",f'S_{student_model_name}_bs{batch_size}_lr{learning_rate}')
os.makedirs(log_path,exist_ok = True)


logfile = os.path.join(log_path, "output.log")
student_save_path=os.path.join(log_path ,'model_Talpha.pt')



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
    transforms.ToTensor()])

train_dataset=Train_Data(Train_Label_Path, Train_Data_Path, H5py_Path, transform=data_transformations, is_multimodal=True) 

train_size = int(r * len(train_dataset)) 
valid_size = len(train_dataset) - train_size 
Train_Dataset,Valid_Dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size]) 

Test_Dataset=Test_Data(Test_Label_Path, Test_Data_Path, H5py_Path,transform=data_transformations, is_multimodal=True)

train_loader=DataLoader(dataset=Train_Dataset,batch_size=batch_size,shuffle=True, drop_last = True) # Create Training Dataloader 
valid_loader=DataLoader(dataset=Valid_Dataset,batch_size=batch_size,shuffle=False, drop_last = True)# Create Validation Dataloader 
test_loader=DataLoader(dataset=Test_Dataset,batch_size=batch_size,shuffle=False, drop_last = True) # Create Test Dataloader 


MLP_model_t = timm.create_model(teacher_model_name, pretrained=True)

MLP_model_s = timm.create_model(student_model_name, pretrained=True)

# print(MLP_model_s)
# sys.exit()
Teacher_Model = basic_fusion(MLP_model_t, Num_Classes, teacher_model_name)

Teacher_Model.load_state_dict(torch.load(teacher_weights_path))
Teacher_Model=Teacher_Model.to(device)



alpha=random.uniform(0, 1)

Student_Model = basic_fusion(MLP_model_s, Num_Classes, student_model_name)
Student_Model = Student_Model.to(device)

logging.info(f"Total number of parameters:\n{teacher_model_name}: {sum(p.numel() for p in Teacher_Model.parameters())},\n{student_model_name}: {sum(p.numel() for p in Student_Model.parameters())}")


alpha_model=ANN(batch_size, Num_Classes)
alpha_model=alpha_model.to(device)


Student_optimizer = optim.Adam(list(Student_Model.parameters())+list(alpha_model.parameters()),lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Define Loss Function 

def WEIEN_Loss(Predicted_Teacher_Label,Predicted_Student_Label,T,y,device,criterion):

        Predicted_Teacher_Label=Predicted_Teacher_Label.mean(dim=1)
        AVG_Prob=F.softmax(Predicted_Teacher_Label/T,dim=1).to(device)
      

        Student_Prob=F.log_softmax(Predicted_Student_Label/T,dim=1)
        Student_Loss=F.kl_div(Student_Prob,AVG_Prob)
    
        return AVG_Prob,Student_Loss

def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc
def train(Teacher_Model,Student_Model,device,iterator, optimizer, criterion,alpha,Temp): # Define Training Function 

    early_stopping = EarlyStopping(patience=7, verbose=True)
    #print("Training Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    alpha_val = 0
    Student_Model.train() # call model object for training 
    with torch.autograd.set_detect_anomaly(True):
        # print(len(iterator),batch_size)
        for (aud,img,y) in iterator:
            aud=aud.float()
            
            img = img.float()
            
            aud=aud.to(device)
            img = img.to(device)        
        
            y = y.type(torch.LongTensor)
            label_duplicate=nn.functional.one_hot(y, num_classes=Num_Classes)
            
            y=y.to(device)# Transfer label  to device
            
            
            concat_arr=[]
     
            optimizer.zero_grad() # Initialize gredients as zeros 
            count=count+1
            
            Predicted_Teacher_Label=Teacher_Model(aud,img)
         
            
            Predicted_Student_Label=Student_Model(aud,img)
  
            CEL_Loss = criterion(Predicted_Student_Label, y)
            AVGprob,EN_AVG_Loss=WEIEN_Loss(Predicted_Teacher_Label,Predicted_Student_Label,Temp,y,device,criterion)
            # print(y.shape,L_prob.shape,Predicted_Student_Label.shape)

            AVGprob=AVGprob.to(device)
            label_duplicate=label_duplicate.to(device)
            s_gt= label_duplicate-AVGprob
            s_gs= label_duplicate-Predicted_Student_Label
            s_ts= AVGprob-Predicted_Student_Label

            # s_gt=s_gt.to(device)
            # s_gs=s_gs.to(device)
            # s_ts=s_ts.to(device)
            
            

            concat_arr=torch.cat((label_duplicate,AVGprob,Predicted_Student_Label,s_gt,s_gs,s_ts),1)
            concat_arr=torch.flatten(concat_arr)

            alpha=alpha_model(concat_arr)
            alpha_val += alpha.detach().cpu().numpy()
            # print(alpha)
            alpha_loss=CEL_Loss

          
            loss=((1-alpha)*(CEL_Loss))+(alpha*(Temp*Temp)*(EN_AVG_Loss))+alpha_loss
            acc = calculate_accuracy(Predicted_Student_Label,y) # training accuracy 
          
            loss.backward(retain_graph=True) # backpropogation 
            optimizer.step() # optimize the model weights using an optimizer 
            epoch_loss += loss.item() # sum of training loss
            epoch_acc += acc.item() # sum of training accuracy 
        # print('epoch done')
        return epoch_loss / len(iterator), epoch_acc / len(iterator) , alpha_val / len(iterator)
def evaluate(model,device,iterator, criterion): # Evaluate Validation accuracy 
    #print("Validation Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    model.eval() # call model object for evaluation 
    
    precision = 0
    recall = 0
    fscore = 0
        
    with torch.no_grad(): # Without computation of gredient 
        for (aud,img,y) in iterator:
            aud=aud.float()
            
            img = img.float()
            
            aud=aud.to(device)
            img = img.to(device)
            y = y.type(torch.LongTensor)
            y=y.to(device)# Transfer label  to device
            
            count=count+1
            Predicted_Label = model(aud,img) # Predict claa label 
          
            loss = criterion(Predicted_Label, y) # Compute Loss 
            acc = calculate_accuracy(Predicted_Label, y) # compute Accuracy 
            #print("Validation Iteration Number=",count)
            epoch_loss += loss.item() # Compute Sum of  Loss 
            epoch_acc += acc.item() # Compute  Sum of Accuracy 
            
            Predicted_Label_2=Predicted_Label.detach().cpu().numpy()
            Predicted_Label_2=np.argmax(Predicted_Label_2,axis=1)
            y_2=y.detach().cpu().numpy()
            precision1, recall1, fscore1, sup = sklearn.metrics.precision_recall_fscore_support(y_2, Predicted_Label_2, average='weighted')

            precision=precision + precision1
            recall=recall+recall1
            fscore=fscore+fscore1

    return epoch_loss / len(iterator), epoch_acc / len(iterator) , precision/len(iterator), recall/len(iterator), fscore/len(iterator)
        

MODEL_SAVE_PATH = student_save_path
best_valid_loss = float('inf')

alpha_values = []

logging.info("Training ...") 

logging.info("---------------------------------------------------------------------------------------------------------------------")   
early_stopping = EarlyStopping(patience=7, verbose=True) # early Stopping Criteria
for epoch in range(EPOCHS):
    start_time=time.time() # Compute Start Time 
    train_loss, train_acc, alpha_val = train(Teacher_Model,Student_Model,device,train_loader,Student_optimizer,criterion,alpha,Temprature) # Call Training Process 
    alpha_values.append(alpha_val)
    
    train_loss=round(train_loss,2) # Round training loss 
    train_acc=round(train_acc,2) # Round training accuracy 
    valid_loss, valid_acc,_,_,_ = evaluate(Student_Model,device,valid_loader,criterion) # Call Validation Process 
    valid_loss=round(valid_loss,2) # Round validation loss
    valid_acc=round(valid_acc,2) # Round accuracy 
    end_time=(time.time()-start_time) # Compute End time 
    end_time=round(end_time,2)  # Round End Time 
    logging.info(f" | Epoch={epoch} | Training Accuracy={train_acc*100} | Validation Accuracy= {valid_acc*100} | Training Loss= {train_loss} | Validation_Loss= {valid_loss} Time Taken(Seconds)={end_time}|")
    logging.info("---------------------------------------------------------------------------------------------------------------------")
    
    early_stopping(valid_loss,Student_Model,MODEL_SAVE_PATH) # call Early Stopping to Prevent Overfitting 
    if early_stopping.early_stop:
        logging.info("Early stopping")
        break
    Student_Model.load_state_dict(torch.load(MODEL_SAVE_PATH))

Student_Model.load_state_dict(torch.load(MODEL_SAVE_PATH)) # load the trained model 
test_loss, test_acc,p,r,f1 = evaluate(Student_Model, device, test_loader, criterion) # Compute Test Accuracy on Unseen Signals 

test_loss=round(test_loss,2)# Round test loss
test_acc=round(test_acc,2) # Round test accuracy

p=round(p,3) 
r=round(r,3) 
f1=round(f1,3)

logging.info(f"|Test Loss= {test_loss} Test Accuracy= {test_acc*100}") # print test accuracy 
logging.info(f"P: {p}, R: {r}, F1: {f1}") 