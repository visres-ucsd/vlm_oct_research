# Basic Imports
from copy import copy
import datetime
import math
import os
import sys
from pathlib import Path
import random
import pickle
import numpy as np
from constants import *
from tqdm.auto import tqdm

# PyTorch related imports....
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import time
from rFmodel import *

# Metrics & tracking....
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from data_loader import *
from focal_loss_imp import *
import pytorch_warmup as warmup
from sklearn.model_selection import KFold
import pandas as pd
import warnings
warnings.filterwarnings('always')


def train_model(model,
                criterion,
                optimizer,
                scheduler,
                num_epochs=25,
                model_save_name = "",
                summary_writer = None):

    # training run statistics...
    run_df = pd.DataFrame(columns= ['train_loss', 'train_acc', 'val_loss', 
                                    'val_acc', 'val_precision_healthy', 'val_recall_healthy', 
                                    'val_precision_glaucoma', 'val_recall_glaucoma', 'val_f1_score', 'val_auc_healthy'])

    since = time.time()
    best_f1_score = 0.0
    best_auc_score = 0.0
    acc_thresh = 0.5



    # defining metric holding variables....
    train_epoch_loss = []
    train_epoch_acc  = []

    val_epoch_loss = []
    val_epoch_acc  = []
    val_epoch_prec_h = []
    val_epoch_prec_g = []
    val_epoch_recall_h = []
    val_epoch_recall_g = []
    val_epoch_f1 = []
    val_epoch_auc_h = []
    patience = 10
    p_cnt = 0
  

    # Training Epochs....
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Training steps.....
        model.train()
        running_loss = 0.0
        running_corrects = 0
        random_batches_train = random.sample(range(len(train_dataloader)), num_train_samples_viz)
        random_collection_train_sample = []
        cnt = 0
        lr_progress = []

        for inputs, labels,_ in tqdm(train_dataloader, position = 0, leave = True):

            # moving tensors to gpu....
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)

                preds = (outputs > acc_thresh)*1
                preds = preds.squeeze(-1)

                loss = criterion(outputs,
                                 torch.unsqueeze(labels,-1).type(torch.float),
                                 alpha_neg = training_data.class_weights[0],
                                 alpha_pos = training_data.class_weights[1],
                                 gamma = focal_weight,
                                 reduction = "mean")

                # applying computed gradients....
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

            if cnt in random_batches_train:
                random_collection_train_sample.append(inputs.detach().cpu()[0])
            cnt+=1

            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warm_lim:
                    scheduler.step()
            
            for param_group in optimizer.param_groups:
                lr_progress.append(param_group['lr'])


        # averaging epoch loss...
        train_epoch_loss.append(running_loss / training_data.total_size)
        train_epoch_acc.append(running_corrects.double() / training_data.total_size)

        # Validation step...
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        running_prec_h = 0.0
        running_prec_g = 0.0
        running_recall_h = 0.0
        running_recall_g = 0.0
        running_f1 = 0.0
        running_auc_h = 0.0

        for inputs, labels,_ in tqdm(valid_dataloader, position = 0, leave = True):
            # moving tensors to gpu....
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                preds = (outputs > acc_thresh)*1
                preds = preds.squeeze(-1)

                loss = criterion(outputs,
                                 torch.unsqueeze(labels,-1).type(torch.float),
                                 alpha_neg = validation_data.class_weights[0],
                                 alpha_pos = validation_data.class_weights[1],
                                 gamma = focal_weight,
                                 reduction = "mean")

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)


            # other metrics class wise computation....
            running_prec_h += precision_score(y_true = labels.detach().cpu().numpy(),
                                              y_pred = preds.detach().cpu().numpy(),
                                              average = "binary",
                                              pos_label = 1,
                                              zero_division = 0.0)

            running_prec_g += precision_score(y_true = labels.detach().cpu().numpy(),
                                              y_pred = preds.detach().cpu().numpy(),
                                              average = "binary",
                                              pos_label = 0,
                                              zero_division = 0.0)

            #######################################################
            running_recall_h += recall_score(y_true = labels.detach().cpu().numpy(),
                                              y_pred = preds.detach().cpu().numpy(),
                                              average = "binary",
                                              pos_label = 1,
                                              zero_division = 0.0)

            running_recall_g += recall_score(y_true = labels.detach().cpu().numpy(),
                                              y_pred = preds.detach().cpu().numpy(),
                                              average = "binary",
                                              pos_label = 0,
                                              zero_division = 0.0)


            running_f1 += (f1_score(y_true = labels.detach().cpu().numpy(),
                                   y_pred = preds.detach().cpu().numpy(),
                                   average = "binary",
                                   pos_label = 0,
                                   zero_division=0) + f1_score(y_true = labels.detach().cpu().numpy(),
                                   y_pred = preds.detach().cpu().numpy(),
                                   average = "binary",
                                   pos_label = 1,
                                   zero_division=0))/2


            #######################################################
            try:
                running_auc_h += roc_auc_score(y_true  = labels.detach().cpu().numpy(),
                                               y_score = outputs.detach().cpu().numpy())
            except:
                running_auc_h += 0


        # Averaging Metric Values for plotting.....
        val_epoch_loss.append(running_loss / validation_data.total_size)
        val_epoch_acc.append(running_corrects.double() / validation_data.total_size)
        val_epoch_prec_h.append(running_prec_h /len(valid_dataloader))
        val_epoch_prec_g.append(running_prec_g /len(valid_dataloader))
        val_epoch_recall_h.append(running_recall_h /len(valid_dataloader))
        val_epoch_recall_g.append(running_recall_g /len(valid_dataloader))
        val_epoch_f1.append(running_f1 / len(valid_dataloader))
        val_epoch_auc_h.append(running_auc_h /len(valid_dataloader))

        # Tensorboard metric plotting....
        summary_writer.add_scalar('Loss/train', train_epoch_loss[-1], epoch)
        summary_writer.add_scalar('Acc/train', train_epoch_acc[-1], epoch)
        summary_writer.add_scalar('Loss/valid', val_epoch_loss[-1], epoch)
        summary_writer.add_scalar('Acc/valid', val_epoch_acc[-1], epoch)
        summary_writer.add_scalar('Healthy Precision/valid', val_epoch_prec_h[-1], epoch)
        summary_writer.add_scalar('Glaucoma Precision/valid', val_epoch_prec_g[-1], epoch)
        summary_writer.add_scalar('Healthy Recall/valid', val_epoch_recall_h[-1], epoch)
        summary_writer.add_scalar('Glaucoma Recall/valid', val_epoch_recall_g[-1], epoch)
        summary_writer.add_scalar('F1 score overall/valid', val_epoch_f1[-1], epoch)
        summary_writer.add_scalar('AUC/valid', val_epoch_auc_h[-1], epoch)
        summary_writer.add_images('Augmented Images/ Train', torch.stack(random_collection_train_sample, dim = 0), epoch)
        summary_writer.add_scalar('Learning Rate Progress', lr_progress[-1], epoch)

        print("Training   loss : {} | Training   Accuracy : {}".format(train_epoch_loss[-1], train_epoch_acc[-1]))
        print("Validation loss : {} | Validation Accuracy : {}".format(val_epoch_loss[-1], val_epoch_acc[-1]))
        print("Learning   Rate : {}".format(lr_progress[-1]))

        if best_auc_score < val_epoch_auc_h[-1]:
            print("Saving the best model with best auc score.....")
            torch.save(model.state_dict(), save_dir_name + model_name + model_save_name + ".pt")
            best_auc_score = val_epoch_auc_h[-1]

            print("Validation Precision -  Healthy : {} | Glaucoma : {}".format(val_epoch_prec_h[-1],   val_epoch_prec_g[-1]))
            print("Validation Recall    -  Healthy : {} | Glaucoma : {}".format(val_epoch_recall_h[-1], val_epoch_recall_g[-1]))
            print("Validation AUC       -  {}".format(val_epoch_auc_h[-1]))
            print("Validation F1 Score  -  {}".format(val_epoch_f1[-1]))

            # zero outing patience
            p_cnt = 0
        
        if epoch != 0:
            if val_epoch_auc_h[-1] <= val_epoch_auc_h[-2]:
                p_cnt+=1


        row_entry = {'train_loss' : train_epoch_loss[-1], 
                     'train_acc'  : train_epoch_acc[-1],
                     'val_loss'   : val_epoch_loss[-1], 
                     'val_acc'    : val_epoch_acc[-1], 
                     'val_precision_healthy'  : val_epoch_prec_h[-1], 
                     'val_recall_healthy'     : val_epoch_recall_h[-1], 
                     'val_precision_glaucoma' : val_epoch_prec_g[-1], 
                     'val_recall_glaucoma'    : val_epoch_recall_g[-1], 
                     'val_f1_score' : val_epoch_f1[-1], 
                     'val_auc_healthy' : val_epoch_auc_h[-1]}
        
        

        # storing stats....
        run_df = pd.concat([run_df, pd.DataFrame(data = row_entry,index=[0])], ignore_index=True)

        if p_cnt == patience:
            print("Early Stopping, saving csv.....")
            run_df.to_csv(save_dir_name + model_name + model_save_name + ".csv")
            return 

    print("Training complete........")
    run_df.to_csv(save_dir_name + model_name + model_save_name + ".csv")
    return



# Impoving code reproducability...
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.benchmark = True

# setting up gpu details
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

print("PyTorch version : {} | Device type : {}".format(torch.__version__, device))


# dataset loading (file path in constants.py file)
# testing is done in a different file....
all_data_df = pd.read_csv(dataset_csv)
# mapping labels to indices for training....
all_data_df["new_labels"] = all_data_df["EyeDX_at_spectralisDate"].apply(lambda x : label_mapping[x])
train_df = all_data_df[all_data_df["Split"] == "train"]
valid_df = all_data_df[all_data_df["Split"] == "val"]
test_df  = all_data_df[all_data_df["Split"] == "test"]

print("#"*30)

print("Total number of train images : ",len(train_df))
print("Total number of valid images : ",len(valid_df))

print("#"*30)
print("Label distribution :")
print("Training split   : Healthy : {} | Glaucoma : {}".format(len(train_df[train_df["new_labels"] == 1]) / len(train_df),
                                                               len(train_df[train_df["new_labels"] == 0]) / len(train_df)))
print("#"*30)
print("Validation split : Healthy : {} | Glaucoma : {}".format(len(valid_df[valid_df["new_labels"] == 1]) / len(valid_df),
                                                               len(valid_df[valid_df["new_labels"] == 0]) / len(valid_df)))
print("#"*30)
print("Testing split : Healthy : {} | Glaucoma : {}".format(len(test_df[test_df["new_labels"] == 1]) / len(test_df),
                                                               len(test_df[test_df["new_labels"] == 0]) / len(test_df)))

print("#"*30)
# Details of model weight saving...
print("Model save name :",model_save_name)
if not os.path.exists(save_dir_name + model_name):
    os.makedirs(save_dir_name + model_name, exist_ok = True)
    os.makedirs(save_dir_name + model_name + "/logs/", exist_ok = True)
    os.makedirs(save_dir_name + model_name + "/logs/scalars/", exist_ok = True)
print("#"*30)

# Setting up tensorboard logging....
logdir = save_dir_name + model_name + "/logs/scalars/" + model_save_name
summary_writer = SummaryWriter(logdir)
    

# image pre-processing functions taken from RetFound as it is...
pre_proc_func = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

# building dataloaders....
training_data = GenerateDataset(image_files = train_df["OCTImagePath"],
                                label_list  = train_df["new_labels"],
                                img_res     = input_shape[0],
                                augment     = use_aug,
                                transform   = pre_proc_func,
                                apply_random_prob = aug_prob,
                                split_flag  = "training")

print("#"*30)
validation_data = GenerateDataset(image_files = valid_df["OCTImagePath"],
                                  label_list  = valid_df["new_labels"],
                                  img_res     = input_shape[0],
                                  augment     = False,
                                  transform = pre_proc_func,
                                  split_flag  = "validation")

train_dataloader = DataLoader(training_data,
                              batch_size = batch_size,
                              shuffle=True,
                              num_workers=4)

valid_dataloader = DataLoader(validation_data,
                              batch_size = batch_size,
                              shuffle=True,
                              num_workers=4)

print("#"*30)
print("Number of Training   steps : ",len(train_dataloader))
print("Number of Validation steps : ",len(valid_dataloader))
print("#"*30)

# Loading the model....
# All model constants are specified in the constants file
# Make experimental changes in it....
base_model = build_model()

# Freezing the base model & only unfreezing the top added layers......
trainable_cnt = 0
total_cnt = 0
num_layers = sum(1 for _ in base_model.named_parameters())
limit = num_layers - int(num_layers*unfreeze_perc)
    
for nm, param in base_model.named_parameters():
    # Freezing classification layer makes no sense..
    total_cnt+=1
    if "layer_1" in nm:
        param.requires_grad = True
        trainable_cnt+=1
        continue

    if total_cnt < limit:
        param.required_grad = False
    else:
        param.requires_grad = True
        trainable_cnt+=1
        

print("#"*30)
print("Percentage of trainabile parameters : {:.2f}".format((trainable_cnt / total_cnt)*100))
base_model = base_model.to(device)


# for binary classification cases.....
criterion=sigmoid_focal_loss

# Observe that all parameters are being optimized (taken fron RetFound Paper)
optimizer_ft = optim.SGD(base_model.parameters(), lr = learning_rate, weight_decay=l2_reg)
num_steps = len(train_dataloader)*train_epochs
warm_lim  = len(train_dataloader)*10  
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max = num_steps - warm_lim, eta_min = last_lr)
warmup_scheduler = warmup.LinearWarmup(optimizer_ft, warm_lim)

print("#"*30)

# Performing Training steps....
model_ft = train_model(model = base_model,
                       criterion  = criterion,
                       optimizer  = optimizer_ft,
                       scheduler  = lr_scheduler,
                       num_epochs = train_epochs,
                       model_save_name = model_save_name,
                       summary_writer = summary_writer)
   
print("########## END OF TRAINING ############")
