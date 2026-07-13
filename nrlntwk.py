import torch
from sklearn.datasets import load_digits
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import statistics
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

digits=load_digits()
X=digits.data
y=digits.target

X_tensor=torch.tensor(X,dtype=torch.float32) #입력 값 type 변환
y_tensor=torch.tensor(y,dtype=torch.long) #정답 값 type 변환

X_train, X_test, y_train, y_test= train_test_split(
    X_tensor,y_tensor, test_size=0.2,random_state=42
)

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(64,32) #첫 뉴런 가중치 및 편향 생성
        self.output= nn.Linear(32,10) #아웃풋 가중치 및 편향 생성
    
    def forward(self,x):
        x=self.hidden(x) #첫 뉴런 계산
        x=torch.relu(x) #relu 계산
        x=self.output(x) #아웃풋 계산
        return x

class DigitClassifierDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1=nn.Linear(64,32)
        self.hidden2=nn.Linear(32,32)
        self.output=nn.Linear(32,10)

    def forward(self,x):
        x=self.hidden1(x)
        x=torch.relu(x)
        x=self.hidden2(x)
        x=torch.relu(x)
        x=self.output(x)
        return x

class DigitClassifierWide(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(64, 32)
        self.hidden2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.hidden2(x)
        x = torch.relu(x)
        x = self.output(x)
        return x
        
class DigitClassifierLeaky(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(64,32) #첫 뉴런 가중치 및 편향 생성
        self.output= nn.Linear(32,10) #아웃풋 가중치 및 편향 생성
    
    def forward(self,x):
        x=self.hidden(x) #첫 뉴런 계산
        x=F.leaky_relu(x) #relu 계산
        x=self.output(x) #아웃풋 계산
        return x

class DigitClassifierDeepLeaky(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(64, 32)
        self.hidden2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 10)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.leaky_relu(x)
        x = self.hidden2(x)
        x = F.leaky_relu(x)
        x = self.output(x)
        return x

class DigitClassifierWideLeaky(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(64, 32)
        self.hidden2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.leaky_relu(x)
        x = self.hidden2(x)
        x = F.leaky_relu(x)
        x = self.output(x)
        return x


model=DigitClassifier()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

train_dataset=TensorDataset(X_train, y_train)
train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True)

def train_and_evaluate(model_class):
    model = model_class()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        predicted = torch.argmax(test_output, dim=1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)

    return accuracy

def run_multiple_trials(model_class, n=20):
    accuracies=[]
    for i in range(n):
        acc=train_and_evaluate(model_class)
        accuracies.append(acc)
        print(f"시도 {i+1}: {acc:.4f}")

    mean_acc=statistics.mean(accuracies)
    std_acc=statistics.stdev(accuracies)
    print(f"평균: {mean_acc:.4f}, 표준편차: {std_acc:.4f}")
    return accuracies

model_final=DigitClassifier()
optimizer_final=torch.optim.Adam(model_final.parameters(),lr=0.001)

for epoch in range(20):
    for batch_X, batch_y in train_loader:
        optimizer_final.zero_grad()
        output=model_final(batch_X)
        loss=criterion(output, batch_y)
        loss.backward()
        optimizer_final.step()

model_final.eval()
with torch.no_grad():
    test_output=model_final(X_test)
    predicted_final=torch.argmax(test_output,dim=1)

print("정확도:", (predicted_final==y_test).sum().item()/len(y_test))

cm=confusion_matrix(y_test.numpy(),predicted_final.numpy())
print(cm)