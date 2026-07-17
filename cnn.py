from sklearn.datasets import load_digits
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import statistics

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.fc=nn.Linear(72,10)

    def forward(self,x):
        x=self.conv(x)
        x=torch.relu(x)
        x=self.pool(x)
        x=x.reshape(x.size(0),-1)
        x=self.fc(x)
        return x

digits=load_digits()
X=digits.data
y=digits.target

X_tensor=torch.tensor(X, dtype=torch.float32)
y_tensor=torch.tensor(y, dtype=torch.long)

X_images=X_tensor.reshape(-1,1,8,8)

X_train_img,X_test_img,y_train_img,y_test_img=train_test_split(
    X_images,y_tensor,test_size=0.2,random_state=42
)

train_dataset_cnn=TensorDataset(X_train_img, y_train_img)
train_loader_cnn=DataLoader(train_dataset_cnn, batch_size=32, shuffle=True)

criterion=nn.CrossEntropyLoss()

def train_and_evaluate_v2(model_class,loader,X_test_data,y_test_data):
    model=model_class()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

    epochs=20
    for epoch in range(epochs):
        for batch_X,batch_y in loader:
            optimizer.zero_grad()
            output=model(batch_X)
            loss=criterion(output,batch_y)
            loss.backward()
            optimizer.step()
        
    model.eval()
    with torch.no_grad():
        test_output=model(X_test_data)
        predicted=torch.argmax(test_output,dim=1)
        accuracy=(predicted==y_test_data).sum().item()/len(y_test_data)

    return accuracy

print("===DigitCNN===")
accuracies_cnn=[]
for i in range(20):
    acc=train_and_evaluate_v2(DigitCNN,train_loader_cnn,X_test_img,y_test_img)
    accuracies_cnn.append(acc)
    print(f" 시도 {i+1}: {acc:.4f}")

mean_cnn=statistics.mean(accuracies_cnn)
std_cnn=statistics.stdev(accuracies_cnn)
print(f"평균: {mean_cnn:.4f}, 표준편차: {std_cnn:.4f}")

# model_cnn_final=DigitCNN()
# optimizer_cnn=torch.optim.Adam(model_cnn_final.parameters(),lr=0.001)

# for epoch in range(20):
#     for batch_X,batch_y in train_loader_cnn:
#         optimizer_cnn.zero_grad()
#         output=model_cnn_final(batch_X)
#         loss=criterion(output,batch_y)
#         loss.backward()
#         optimizer_cnn.step()

# model_cnn_final.eval()
# with torch.no_grad():
#     test_output=model_cnn_final(X_test_img)
#     predicted=torch.argmax(test_output,dim=1)
#     accuracy=(predicted==y_test_img).sum().item()/len(y_test_img)

# print("정확도:",accuracy)

# filters = model_cnn_final.conv.weight.detach()
# print(filters.shape)

# fig, axes = plt.subplots(2, 4, figsize=(10, 5))

# for i in range(8):
#     row = i // 4
#     col = i % 4
#     axes[row, col].imshow(filters[i][0], cmap='gray')
#     axes[row, col].set_title(f"filter {i}")
#     axes[row, col].axis('off')

# plt.tight_layout()
# plt.show()
