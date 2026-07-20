import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import time
import json

class FashionClassifier(nn.Module):
    def __init__(self,hidden_size=32):
        super().__init__()
        self.hidden = nn.Linear(784,hidden_size) #첫 뉴런 가중치 및 편향 생성
        self.output= nn.Linear(hidden_size,10) #아웃풋 가중치 및 편향 생성
    
    def forward(self,x):
        x=self.hidden(x) #첫 뉴런 계산
        x=torch.relu(x) #relu 계산
        x=self.output(x) #아웃풋 계산
        return x

class FashionClassifierDeep(nn.Module):
    def __init__(self,hidden_size=32):
        super().__init__()
        self.hidden1=nn.Linear(784,hidden_size)
        self.hidden2=nn.Linear(hidden_size,hidden_size)
        self.output=nn.Linear(hidden_size,10)

    def forward(self,x):
        x=self.hidden1(x)
        x=torch.relu(x)
        x=self.hidden2(x)
        x=torch.relu(x)
        x=self.output(x)
        return x

def train_and_evaluate(model_fn,train_loader,test_loader, epochs=20):
    model=model_fn()
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

    for epoch in range(epochs):
        for images, labels in train_loader:
            images=images.view(images.size(0),-1)
            optimizer.zero_grad()
            output=model(images)
            loss=criterion(output,labels)
            loss.backward()
            optimizer.step()

    correct=0
    total=0
    model.eval()
    with torch.no_grad():
        for images,labels in test_loader:
            images=images.view(images.size(0),-1)
            output=model(images)
            _,predicted=torch.max(output,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    
    return correct/total

transform=transforms.ToTensor()


train_data=torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data=torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

batch_size=64

train_loader=DataLoader(train_data, batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=False)

model=FashionClassifier()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

configs={
    "single_32":lambda:FashionClassifier(hidden_size=32),
    "single_64": lambda:FashionClassifier(hidden_size=64),
    "deep_32_32":lambda:FashionClassifier(),
    "deep_64_64":lambda:FashionClassifier(hidden_size=64),
}


results={name:[]for name in configs}
start=time.time()

for name,model_fn in configs.items():
    for i in range(5):
        acc=train_and_evaluate(model_fn,train_loader,test_loader)
        results[name].append(acc)
        print(name,i+1,acc)

print("총 소요 시간:", time.time()-start,"sec")

for name,accs in results.items():
    accs=np.array(accs)
    se=accs.std()/np.sqrt(len(accs))
    print(name,"평균:",accs.mean(),"표준편차:",accs.std(),"표준오차:",se)

with open("results.json","w") as f:
    json.dump(results,f)

print("저장 완료")