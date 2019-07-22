# 붓꽃의 품종 분류
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
import numpy


iris_dataset = load_iris()
#
# print("iris_dataset 의 키:\n", iris_dataset.keys())
#
# print(iris_dataset['DESCR'][:193] + "\n...")
#
# print("타깃의 이름:", iris_dataset["target_names"])
# print("특성의 이름:", iris_dataset["feature_names"])
# print("data 의 타입:", type(iris_dataset["data"]))
#
# print("data 의 크기:", iris_dataset["data"].shape)
#
# print("data 의 처음 다섯 행:\n", iris_dataset["data"][:5])
#
# print("target 의 타입",  type(iris_dataset["target"]))
# print("target 의 크기", iris_dataset["target"].shape)
# print("target:\n", iris_dataset["target"])


x_train, x_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)

print("X TRAIN 사이즈:", x_train.shape )
print("y TRAIN 사이즈:", y_train.shape )
print("X TEST 사이즈:", x_test.shape )
print("y TEST 사이즈:", y_test.shape )

iris_dataframe = pd.DataFrame(x_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='0', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

plt.show()