from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import mglearn

cancer = load_breast_cancer()
print("cancer.keys():\n", cancer.keys())

print(cancer.data.shape)

print("클래스별 샘플 개수:\n", {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})

print("특성 이름:\n", cancer.feature_names)

boston = load_boston()
print("데이터의 형태", boston.data.shape)

X, y = mglearn.datasets.load_extended_boston()
print(X.shape)

mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# plt.show()

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

print("테스트 세트 예측:", clf.predict(X_test))
print("테스트 세트 정확도: {:.2f}".format(clf.score(X_test, y_test)))

font = {"family": "NanumGothic"}
plt.rc("font", family="NanumGothic")

fig, axes = plt.subplots(1, 3, figsize=(10,3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} 이웃".format(n_neighbors))
    ax.set_xlabel("특성 0", fontdict=font)
    ax.set_ylabel("특성 1", fontdict=font)

axes[0].legend(loc=3)
# plt.show()