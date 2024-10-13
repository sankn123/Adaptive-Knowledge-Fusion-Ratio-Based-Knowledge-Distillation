import numpy as np 
import torch 
import timm

import sys
sys.path.append("/export/home/vivian/sankn/AdaptiveKD/training_scripts")
from utils.custom_dataloaders import *
from utils.pytorchtools import EarlyStopping
from models.teacher import Teacher
from models.student import Student
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

# dataset = "FSC22"
# dataset = "DCASE19"
dataset = "ESC10"


# student_model_name = "vit_small_patch16_224"
# student_model_name = "vit_tiny_patch16_224"
# student_model_name = "resnet152"
# student_model_name = "resnet50"
student_model_name = "resnet18"

is_student_pretrained = True

teacher_model_name = "vit_base_patch16_224"


Temprature=200
plot_alpha = True

match dataset:
    case "FSC22":
        from utils.paths.FSC22_pathname import *
        Num_Classes = 27
        r = 0.9 # train/valid split
        learning_rate=1e-4
        batch_size=16

    case "DCASE19":
        from utils.paths.DCASE19_pathname import *
        Num_Classes = 10       
        r = 0.9 # train/valid split
        learning_rate=2e-3
        batch_size=32

    case "ESC10":
        from utils.paths.ESC10_pathname import *
        Num_Classes = 10
        r = 0.9 # train/valid split
        learning_rate=2e-4
        batch_size=16
        
    case _:
        print(f"{dataset} Not implemented")
        sys.exit()


log_path = os.path.join(save_path,"weighted_AKD",f'S_{student_model_name}_bs{batch_size}_lr{learning_rate}')
os.makedirs(log_path,exist_ok = True)


logfile = os.path.join(log_path, "output.log")
student_save_path=os.path.join(log_path ,'model_Talpha.pt')
teacher_weights_path = os.path.join("/export/home/vivian/sankn/AdaptiveKD/teacher_model_weights",dataset,"model.pt")
# log_path2 = os.path.join(save_path,"weighted_AKD",f'S_{student_model_name}_bs{batch_size}_lr{learning_rate}',"plots")


log_path = os.path.join(save_path,"weighted_AKD",f'S_{student_model_name}_bs{batch_size}_lr{learning_rate}',"plots")
os.makedirs(log_path,exist_ok = True)

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


Test_Dataset=Test_Data(Test_Label_Path, Test_Data_Path, H5py_Path,transform=data_transformations)

test_loader=DataLoader(dataset=Test_Dataset,batch_size=1,shuffle=False, drop_last = True) # Create Test Dataloader 


MLP_model = timm.create_model(teacher_model_name, pretrained=True)

Teacher_Model=Teacher(MLP_model,Num_Classes)

Teacher_Model.load_state_dict(torch.load(teacher_weights_path))
Teacher_Model=Teacher_Model.to(device)


# alpha=random.uniform(0, 1)


Student_Model=Student(student_model_name, Num_Classes, is_student_pretrained)
Student_Model=Student_Model.to(device)

Student_Model2=Student(student_model_name, Num_Classes, is_student_pretrained)
Student_Model2=Student_Model2.to(device)

MODEL_SAVE_PATH = student_save_path
Student_Model.load_state_dict(torch.load("/export/home/vivian/sankn/AdaptiveKD/logs/ESC10/AKD/S_resnet18_bs16_lr0.0002/model_Talpha.pt")) # load the trained model 

Student_Model2.load_state_dict(torch.load(MODEL_SAVE_PATH)) # load the trained model 


def Patch_Weights(Label,Actual_label,T):
    #print("Predicted shape=",Label.shape)
    #print("Actual shape=",Actual_label.shape)
    #print("Actual Lable=",Actual_label)
    criterion = nn.CrossEntropyLoss() 
    #print(Actual_label)
    [B,C,W]=Label.shape 
    #print(B,C,W)
    rows=B
    cols=Num_Classes
    Weighted_prob=torch.zeros([rows,cols]).cuda()
    for i in range(0,B):
        W=torch.zeros([C,1]).cuda()
        Teacher_Prob=torch.zeros([C,Num_Classes]).cuda()
        for j in range(0,C):
            
            Temp1=Label[i,j].unsqueeze(0)
            #Temp1=Label[i,j]
            #print(Temp1)
            y=Actual_label[i].unsqueeze(0)
            #y=Actual_label[i]
            #print(Temp1.shape)
            #print(y.shape)
            Teacher_Prob[j]=Label[i,j]
            #print(Temp1.shape)
            W[j]=criterion(Temp1,y)
            Temp1=Temp1.squeeze(0).detach()
            y=y.squeeze(0)
            y=y.detach()
            #print(W[j])
        W=1-torch.exp(W/torch.sum(W))
        #print(W)
        #print("\n")
        #print(Teacher_Prob)
        for k in range(0,C):
            #print(Teacher_Prob[k])
            #print(W[k])
            #Teacher_Prob[k]=torch.mul(Teacher_Prob[k],W[k])
            #if k==1:
                #print(W[k])
                #print(Teacher_Prob[k])
            Teacher_Prob[k]=torch.clone(Teacher_Prob[k])*torch.clone(W[k])
            #print(Teacher_Prob[k])
        #Temp2=W*Teacher_Prob
        Weighted_prob[i]=torch.sum(Teacher_Prob,dim=0)
        Teacher_Prob=Teacher_Prob.detach()
        W=W.detach()
        
        #print(Weighted_prob[i])
   
    # Teacher_Prob=Teacher_Prob.detach()    
    
    #Teacher_Prob=Teacher_Prob.cpu()
    #W=W.cpu()
    #Temp1=Temp1.cpu()
    #y=y.cpu()
    #Weighted_prob=F.softmax(Weighted_prob/T,dim=1)
    # print(Teacher_Prob.shape)
    
    return Weighted_prob, W, Teacher_Prob

def WEIEN_Loss(Predicted_Teacher_Label,Predicted_Student_Label,T,y,device):
        
        #AVG_Prob=Weight_Calculate(Predicted_Teacher_Label,y,criterion,T)
        L_Prob, w, Teacher_Prob = Patch_Weights(Predicted_Teacher_Label,y,T)
        
        # # print(L_Prob.shape)
        Predicted_Teacher_Label=Predicted_Teacher_Label.mean(dim=1)
        G_Prob=F.softmax(Predicted_Teacher_Label/T,dim=1).to(device)
        L_Prob=F.softmax(L_Prob/T,dim=1).to(device)
        AVG_Prob=(L_Prob+G_Prob)/2
        # #print(AVG_Prob)
        # #print(y)
        # Student_Prob=F.log_softmax(Predicted_Student_Label/T,dim=1)
        # Student_Loss=F.kl_div(Student_Prob,AVG_Prob)
        # #print(Total_E_Loss)
        return L_Prob, w, Teacher_Prob
    
def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc




def evaluate(Teacher_Model,Student_Model,device,iterator,Temp): # Evaluate Validation accuracy 
    #print("Validation Starts")

    Teacher_Model.eval() # call model object for evaluation 
    Student_Model.eval()
    mismatch = True
    y_done = []
    with torch.no_grad(): # Without computation of gredient 
        for (x,y) in iterator:
          
            x=x.float()
            #y=y.float()
            x=torch.cat([x,x,x],dim=1)
            #x=ImageToPatches(x,16)
            #print(x.shape)
            x=x.to(device)
            y=y.to(device)# Transfer label  to devic
            
            Predicted_Teacher_Label=Teacher_Model(x)
            #Predicted_Teacher_Label=x = Predicted_Teacher_Label.mean(dim=1)
            Predicted_Student_Label=Student_Model(x)
            
            Predicted_Student_Label2=Student_Model2(x)
            
            
            L_prob, w, Teacher_Prob = WEIEN_Loss(Predicted_Teacher_Label,Predicted_Student_Label,Temp,y,device)
            
            # print("Predicted Teacher Label-mean",torch.argmax(Predicted_Teacher_Label.mean(dim=1)))
            # print("Predicted Student Label",torch.argmax(Predicted_Student_Label),"\nActual Label", y)
            
            
            
            # # print(y.shape,L_prob.shape,Predicted_Student_Label.shape)
            # print("Predicted Teacher Label-weighted",torch.argmax(L_prob),"\nW max index", torch.argmax(w),"\nW max value", torch.max(w))
            # print("-"*10)
            
            if y==torch.argmax(Predicted_Teacher_Label.mean(dim=1)):
                
                if set(range(Num_Classes)).issubset(y_done):
                    return
                
                if y in y_done:
                    continue
                else:
                    y_done.append(int(y.detach().cpu().numpy()))
                    # y_done = np.array(y_done)
                    print(y_done)
                    
                    
                        
                    
                    
                    teach = Predicted_Teacher_Label.mean(dim=1)
                    teach = torch.nn.functional.softmax(teach,1).cpu().numpy().flatten()
                    
                    stu = Predicted_Student_Label
                    stu = torch.nn.functional.softmax(stu,1).cpu().numpy().flatten()
                    
                    stu2 = Predicted_Student_Label2
                    stu2 = torch.nn.functional.softmax(stu2,1).cpu().numpy().flatten()
                    
                    W_t = w
                    w = torch.nn.functional.softmax(w,1).cpu().numpy().flatten()
                    

                    teach_w = []
                    for val in Teacher_Prob.mean(dim=0):
                        teach_w.append(-1*val)
                    
                    teach_w = torch.stack(teach_w)   
                    teach_w = torch.unsqueeze(teach_w,0)
                    
                    teach_w = torch.nn.functional.softmax(teach_w,1).cpu().numpy().flatten()

                    
                    x = np.arange(1, 11)

                    # Plot the bar chart
                    # print(teach)
                    plt.bar(x, teach, color='orange',width=0.5)
                    plt.axhline(0, color='black', linewidth=1.5)  # Add a horizontal line at y=0

                    # Add labels
                    plt.xlabel('Classes')
                    plt.ylabel('Logits')

                    # Add a title if needed
                    plt.title('Predicted Teacher Labels(Avg)')
                    os.makedirs(os.path.join(log_path,"Teacher_plots"),exist_ok = True)
                    plt.savefig(os.path.join(log_path,"Teacher_plots", f"mean_teacher_class_prob{int(y.detach().cpu().numpy())+1}.pdf"), format='pdf')
                    
                    plt.clf()
                    
                    
                    
                    teach = Predicted_Teacher_Label.mean(dim=1).cpu().numpy().flatten()
                    
                    plt.bar(x, teach, color='orange',width=0.5)
                    plt.axhline(0, color='black', linewidth=1.5)  # Add a horizontal line at y=0

                    # Add labels
                    plt.xlabel('Classes')
                    plt.ylabel('Logits')

                    # Add a title if needed
                    plt.title('Predicted Teacher Labels(Avg)')
                    
                    plt.savefig(os.path.join(log_path,"Teacher_plots", f"mean_teacher_class_{int(y.detach().cpu().numpy())+1}.pdf"), format='pdf')
                    
                    plt.clf()
                    
                    ###
                    
                    plt.bar(x, teach_w, color='orange',width=0.5)
                    plt.axhline(0, color='black', linewidth=1.5)  # Add a horizontal line at y=0

                    # Add labels
                    plt.xlabel('Classes')
                    plt.ylabel('Logits')

                    # Add a title if needed
                    plt.title('Predicted Teacher Labels(weighted)')
                    
                    plt.savefig(os.path.join(log_path, "Teacher_plots" ,f"w_teacher_class_prob{int(y.detach().cpu().numpy())+1}.pdf"), format='pdf')
                    
                    plt.clf()
                    
                    
                    
                    teach_w = []
                    for val in Teacher_Prob.mean(dim=0):
                        teach_w.append(-1*val)
                    
                    teach_w = torch.stack(teach_w) 
                    teach_w = torch.unsqueeze(teach_w,0).cpu().numpy().flatten()
                    
                    plt.bar(x, teach_w, color='orange',width=0.5)
                    plt.axhline(0, color='black', linewidth=1.5)  # Add a horizontal line at y=0

                    # Add labels
                    plt.xlabel('Classes')
                    plt.ylabel('Logits')

                    # Add a title if needed
                    plt.title('Predicted Teacher Labels(weighted)')
                    
                    plt.savefig(os.path.join(log_path, "Teacher_plots",f"w_teacher_class_{int(y.detach().cpu().numpy())+1}.pdf"), format='pdf')
                    
                    plt.clf()
                    
                    
                    ####
                    
                    plt.bar(x, stu, color='orange',width=0.5)
                    plt.axhline(0, color='black', linewidth=1.5)  # Add a horizontal line at y=0

                    # Add labels
                    plt.xlabel('Classes')
                    plt.ylabel('Logits')

                    # Add a title if needed
                    plt.title('Predicted Student Labels using weighted ensemble KD')
                    os.makedirs(os.path.join(log_path,"Student_plots"),exist_ok = True)
                    plt.savefig(os.path.join(log_path, "Student_plots", f"student_class_prob{int(y.detach().cpu().numpy())+1}.pdf"), format='pdf')
                    
                    plt.clf()
                    
                   
                    stu = Predicted_Student_Label.cpu().numpy().flatten()
                    
                    plt.bar(x, stu, color='orange',width=0.5)
                    plt.axhline(0, color='black', linewidth=1.5)  # Add a horizontal line at y=0

                    # Add labels
                    plt.xlabel('Classes')
                    plt.ylabel('Logits')

                    # Add a title if needed
                    plt.title('Predicted Student Labels using weighted ensemble KD')
                    
                    plt.savefig(os.path.join(log_path, "Student_plots", f"student_class_{int(y.detach().cpu().numpy())+1}.pdf"), format='pdf')
                    
                    plt.clf()
                    
                    ##
                    
                    ####
                    
                    plt.bar(x, stu2, color='orange',width=0.5)
                    plt.axhline(0, color='black', linewidth=1.5)  # Add a horizontal line at y=0

                    # Add labels
                    plt.xlabel('Classes')
                    plt.ylabel('Logits')

                    # Add a title if needed
                    plt.title('Predicted Student Labels using average ensemble KD')
                    
                    plt.savefig(os.path.join(log_path, "Student_plots", f"student_class_prob{int(y.detach().cpu().numpy())+1}.pdf"), format='pdf')
                    
                    plt.clf()
                    
                   
                    stu2 = Predicted_Student_Label2.cpu().numpy().flatten()
                    
                    plt.bar(x, stu2, color='orange',width=0.5)
                    plt.axhline(0, color='black', linewidth=1.5)  # Add a horizontal line at y=0

                    # Add labels
                    plt.xlabel('Classes')
                    plt.ylabel('Logits')

                    # Add a title if needed
                    plt.title('Predicted Student Labels using average ensemble KD')
                    
                    plt.savefig(os.path.join(log_path, "Student_plots", f"student_class_{int(y.detach().cpu().numpy())+1}.pdf"), format='pdf')
                    
                    plt.clf()
                    
                    
                    ## 
                    x = np.arange(1, 197)
                    plt.bar(x, w, color='orange',width=0.5)
                    plt.axhline(0, color='black', linewidth=1.5)  # Add a horizontal line at y=0

                    # Add labels
                    plt.xlabel('Dimension')
                    plt.ylabel('Weights')

                    # Add a title if needed
                    plt.title('Patch Weights')
                    os.makedirs(os.path.join(log_path,"patch_weights_plot"),exist_ok = True)
                    plt.savefig(os.path.join(log_path, "patch_weights_plot", f"w_class_prob{int(y.detach().cpu().numpy())+1}.pdf"), format='pdf')
                    
                    plt.clf()
                    
                   
                    w = W_t.cpu().numpy().flatten()
                    
                    plt.bar(x, w, color='orange',width=0.5)
                    plt.axhline(0, color='black', linewidth=1.5)  # Add a horizontal line at y=0

                    # Add labels
                    plt.xlabel('Dimension')
                    plt.ylabel('Weights')

                    # Add a title if needed
                    plt.title('Patch Weights')
                    
                    plt.savefig(os.path.join(log_path, "patch_weights_plot", f"w_class_{int(y.detach().cpu().numpy())+1}.pdf"), format='pdf')
                    
                    plt.clf()
                    
            if mismatch:
                if y==torch.argmax(Predicted_Student_Label) and y!=torch.argmax(Predicted_Student_Label2):
                
                    print("diff")
                    x = np.arange(1, 11)
                    
                    stu_w = torch.nn.functional.softmax(Predicted_Student_Label,1).cpu().numpy().flatten()
                    stu_a = torch.nn.functional.softmax(Predicted_Student_Label2,1).cpu().numpy().flatten()
                    
                    plt.bar(x, stu_w, color='orange',width=0.5)
                    plt.axhline(0, color='black', linewidth=1.5)  # Add a horizontal line at y=0

                    # Add labels
                    plt.xlabel('Classes')
                    plt.ylabel('Logits')

                    # Add a title if needed
                    plt.title('Predicted Student Labels using weighted ensemble KD')
                    os.makedirs(os.path.join(log_path,"mismatch"),exist_ok = True)
                    plt.savefig(os.path.join(log_path,"mismatch", f"weight_vs_avg_student_class_prob_y{int(y.detach().cpu().numpy())+1}_w{int(np.argmax(stu_w))+1}.pdf"), format='pdf')
                    
                    plt.clf()
                    
                   
                    stu_w = Predicted_Student_Label.cpu().numpy().flatten()
                    
                    plt.bar(x, stu_w, color='orange',width=0.5)
                    plt.axhline(0, color='black', linewidth=1.5)  # Add a horizontal line at y=0

                    # Add labels
                    plt.xlabel('Classes')
                    plt.ylabel('Logits')

                    # Add a title if needed
                    plt.title('Predicted Student Labels using weighted ensemble KD')
                    
                    plt.savefig(os.path.join(log_path,"mismatch", f"weight_vs_avg_student_class_y{int(y.detach().cpu().numpy())+1}_w{int(np.argmax(stu_w))+1}.pdf"), format='pdf')
                    
                    plt.clf()
                    
                    ######3
                    
                    plt.bar(x, stu_a, color='orange',width=0.5)
                    plt.axhline(0, color='black', linewidth=1.5)  # Add a horizontal line at y=0

                    # Add labels
                    plt.xlabel('Classes')
                    plt.ylabel('Logits')

                    # Add a title if needed
                    plt.title('Predicted Student Labels using average ensemble KD')
                    
                    plt.savefig(os.path.join(log_path,"mismatch", f"weight_vs_avg_student_class_prob_y{int(y.detach().cpu().numpy())+1}_a{int(np.argmax(stu_a))+1}.pdf"), format='pdf')
                    
                    plt.clf()
                    
                   
                    stu_a = Predicted_Student_Label2.cpu().numpy().flatten()
                    
                    plt.bar(x, stu_a, color='orange',width=0.5)
                    plt.axhline(0, color='black', linewidth=1.5)  # Add a horizontal line at y=0

                    # Add labels
                    plt.xlabel('Classes')
                    plt.ylabel('Logits')

                    # Add a title if needed
                    plt.title('Predicted Student Labels using average ensemble KD')
                    
                    plt.savefig(os.path.join(log_path,"mismatch", f"weight_vs_avg_student_class_y{int(y.detach().cpu().numpy())+1}_a{int(np.argmax(stu_a))+1}.pdf"), format='pdf')
                    
                    plt.clf()
                    mismatch = False
                    
                    
                    

                    
                    
                
    return
        

evaluate(Teacher_Model,Student_Model,device,test_loader,Temprature) # Compute Test Accuracy on Unseen Signals 



