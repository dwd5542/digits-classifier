import torch
from sklearn.datasets import load_digits
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

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

model=DigitClassifier()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

train_dataset=TensorDataset(X_train, y_train)
train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True)

epochs=20

for epoch in range(epochs):

    total_loss=0

    for batch_X,batch_y in train_loader:
        optimizer.zero_grad()

        output=model(batch_X)
        loss=criterion(output, batch_y)

        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
    
    avg_loss=total_loss/len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, 평균 loss: {avg_loss:.4f}")

model.eval()

with torch.no_grad():
    test_output = model(X_test)
    predicted = torch.argmax(test_output, dim=1)
    correct = (predicted == y_test).sum().item()
    accuracy = correct / len(y_test)

print(f"테스트 정확도: {accuracy:.4f}")
