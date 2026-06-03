import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.relu = nn.ReLU(True)
        self.layer_1 = nn.Linear(10, 256)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.layer_2 = nn.Linear(256, 128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.dropout_1 = nn.Dropout(p=0.)
        self.layer_3 = nn.Linear(128, 64)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.dropout_2 = nn.Dropout(p=0.1)
        self.layer_4  = nn.Linear(64, 32)
        self.batchnorm4 = nn.BatchNorm1d(32)
        self.dropout_3 = nn.Dropout(p=0.002)
        self.layer_5 = nn.Linear(32, 10)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs): 
        x = self.layer_1(inputs)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.dropout_1(x)
        x = self.relu(x)
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.dropout_2(x)
        x = self.relu(x)
        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.dropout_3(x)
        x = self.relu(x)
        x = self.layer_5(x)
        return F.log_softmax(x, dim=1)

class ReducedMLP(nn.Module):
    def __init__(self):
        super(ReducedMLP, self).__init__()
        self.relu = nn.ReLU(True)
        self.layer_1 = nn.Linear(10, 64)  # Reduced width from 128 to 64
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.dropout_1 = nn.Dropout(p=0.5)  # Increased dropout probability from 0.2 to 0.5
        self.layer_2 = nn.Linear(64, 32)  # Reduced width from 64 to 32
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.layer_3 = nn.Linear(32, 10)  # Reduced width from 32 to 10
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs): 
        x = self.layer_1(inputs)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout_1(x)
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.layer_3(x)
        return F.log_softmax(x, dim=1)
class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(10, 2000)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2000, 1500)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(1500, 800)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(800, 400)
        self.relu4 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc5 = nn.Linear(400, 150)
        self.relu5 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.2)
        self.fc6 = nn.Linear(150, 12)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.dropout1(x)
        x = self.relu3(self.fc3(x))
        x = self.dropout2(x)
        x = self.relu4(self.fc4(x))
        x = self.dropout3(x)
        x = self.relu5(self.fc5(x))
        x = self.dropout4(x)
        x = self.fc6(x)
        return self.softmax(x)    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))  # or: x.view(x.size(0), -1), x.size(0) = batch_size
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))  # or: x.view(x.size(0), -1), x.size(0) = batch_size
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

