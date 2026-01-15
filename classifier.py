import time
import sys
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

torch.manual_seed(42)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
print(f"GPU used --> {torch.cuda.get_device_name(0)}")

df = pd.read_csv("fashion-mnist_train.csv")


X = df.iloc[:, 1:].values
y = df.iloc[:, 0 ].values


X = X/255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype = torch.float32)
        self.labels   = torch.tensor(labels  , dtype = torch.long)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self,index):
        return self.features[index], self.labels[index]
    

train_dataset = CustomDataset(X_train,y_train)
test_dataset = CustomDataset(X_test,y_test)

batchSize = 32
trainLoader = DataLoader(train_dataset, batch_size = batchSize, shuffle = True)
testLoader  = DataLoader(test_dataset , batch_size = batchSize, shuffle = False)

class myNN(nn.Module):
    
    def __init__(self,num_features):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward(self, x):
        return self.model(x)
    

epochs = 100
learning_rate = 0.1

model = myNN(X_train.shape[1])
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(),lr = learning_rate)



def render(lines):
    sys.stdout.write("\033[H")   
    sys.stdout.write("\033[J")   
    sys.stdout.write("\n".join(lines))
    sys.stdout.flush()

for t in range(5, -1, -1):
    render([
        "+-----------------------------------------+",
       f"|......Training beginning in : {t} sec .....|",
        "+-----------------------------------------+"
    ])
    time.sleep(1)


print()
startTrain = time.time()
for epoch in range(epochs):
    total_epoch_loss = 0
    start = time.time()
    for batch_features, batch_labels in trainLoader:

        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

        output = model(batch_features)
        loss = criterion(output, batch_labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_epoch_loss += loss.item()
    end = time.time()
    avg_loss = total_epoch_loss/len(trainLoader)
    tame = end-start
    print(f"Epoch: [{epoch+1}]  Loss: [{avg_loss:.3f}] Time Taken: {tame} sec")
endTrain = time.time()

timeTrain = endTrain-startTrain
model.eval()

total = 0
correct = 0

with torch.no_grad():
    for batch_features, batch_labels in testLoader:
        
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

        output = model(batch_features)

        _, predicted = torch.max(output, 1)

        total += batch_labels.shape[0]

        correct += (predicted == batch_labels).sum().item()
    print()
    print( "==============================================================================")
    print(f"Accuracy: {(correct/total)*100:.2f}% Total Time taken training: {timeTrain} sec")
    print( "==============================================================================")