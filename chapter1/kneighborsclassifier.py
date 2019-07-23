from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import numpy as np


iris_dataset = load_iris()

knn = KNeighborsClassifier(n_neighbors=1)

# x_train, x_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)
r = knn.fit(X_train, y_train)

print(r)

X_new = np.array([[5, 2.9, 1, 0.2]])

print("X_new.shape", X_new.shape)

## 예측하기
prediction = knn.predict(X_new)
print("예측", prediction)
print("예측한 타깃의 이름:", iris_dataset["target_names"][prediction])

## 평가하기
y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값:\n", y_pred)

## 정확도
print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred == y_test)))
print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test, y_test)))