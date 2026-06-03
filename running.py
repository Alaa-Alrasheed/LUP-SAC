
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from attacks import *
import tools
import numpy as np
import time
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
from tqdm import tqdm
from models import CNN, ResNet18, CifarCNN, RNNClassifier, MLP
from data_loader import get_dataset
from models import CNN, ResNet18, CifarCNN, RNNClassifier, MLP,ReducedMLP,TorchModel
from aggregators import aggregator
import copy
def benignWorker(model,optimizer, train_loader, device,args,idx):
    
    device = args.device
    attack = args.attack
    model.train()
   
    criterion = nn.CrossEntropyLoss()
    print("================ Client # "+str(idx)+" ===================")
    for epoch in range(args.local_iter):
        for images, labels in train_loader:
            #images, labels = next(train_loader)
            images, labels = images.to(device), labels.to(device)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Local Epoch [{epoch+1}/{10}]], Loss: {loss.item():.4f}")
    user_grad,user_grad_org = tools.get_gradient_values(model)

    return user_grad, loss.item(),user_grad_org,model



def byzantineWorker(model,optimizer, train_loader, args,idx):
    device = args.device
    attack = args.attack
    model.train()
   
    criterion = nn.CrossEntropyLoss()
    print("================ Client # "+str(idx)+" ===================")
    for epoch in range(args.local_iter):
        for images, labels in train_loader:
            #images, labels = next(train_loader)
            images, labels = images.to(device), labels.to(device)
            if attack=='label_flip' and args.dataset == "AGNews":
                labels = 3 - labels
            elif attack=='label_flip':
                labels = 9 - labels
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Local Epoch [{epoch+1}/{10}]], Loss: {loss.item():.4f}")
    user_grad,user_grad_org = tools.get_gradient_values(model)

    return user_grad, loss.item(),user_grad_org,model

# define model testing function
def test_classification(epoch,device, model, test_loader, loss_fn, run_time,output_csv="results.csv"):
    model.eval()
    correct = 0
    total = len(test_loader.dataset)
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    acc = 100.0 * correct / total

    # Calculate precision, recall, and F1-score
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    # Calculate the overall loss
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    
    # Create a DataFrame to save the results
    results_df = pd.DataFrame({'Epoch#': [epoch],'Accuracy': [acc], 'Precision': [precision], 'Recall': [recall], 'F1-Score': [f1], 'Loss': [avg_loss],"Time": run_time})
    
    # Save results to CSV
    if epoch ==0:
        results_df.to_csv(output_csv, index=False,mode='a')
    else:
        results_df.to_csv(output_csv, header=False,index=False,mode='a')

    return acc