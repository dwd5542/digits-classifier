import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import json
from torch.utils.data import DataLoader

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.fc=nn.Linear(1352,10)
    
    def forward(self,x):
        x=self.conv(x)
        x=torch.relu(x)
        x=self.pool(x)
        x=x.reshape(x.size(0),-1)
        x=self.fc(x)
        return x

def train_and_evaluate_v2(model_fn,train_loader,test_loader, epochs=20):
    model=model_fn()
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

    for epoch in range(epochs):
        for images, labels in train_loader:
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


configs={
    "fashioncnn":lambda:FashionCNN(),
}


results={name:[]for name in configs}
start=time.time()

for name,model_fn in configs.items():
    for i in range(5):
        acc=train_and_evaluate_v2(model_fn,train_loader,test_loader)
        results[name].append(acc)
        print(name,i+1,acc)

print("총 소요 시간:", time.time()-start,"sec")

for name,accs in results.items():
    accs=np.array(accs)
    se=accs.std()/np.sqrt(len(accs))
    print(name,"평균:",accs.mean(),"표준편차:",accs.std(),"표준오차:",se)

with open("resultsfasioncnn.json","w") as f:
    json.dump(results,f)

print("저장 완료")
