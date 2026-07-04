from sklearn.datasets import load_digits

digits=load_digits()

print("데이터 모양:", digits.data.shape)
print("정답 종류:", digits.target_names)
print("\n첫 번쨰 데이터 (숫자 64개):")
print(digits.data[0])
print("첫 번쨰 정답:", digits.target[0])

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test=train_test_split(
    digits.data, digits.target, test_size=0.2 ,random_state=42
)

model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)

predictions=model.predict(X_test)
accuracy=accuracy_score(y_test, predictions)
print("정확도:", accuracy)

scores=cross_val_score(model,digits.data, digits.target,cv=5)
print("교차 검증 평균:",scores.mean())

cm=confusion_matrix(y_test, predictions)
print("혼동 행렬:")
print(cm)