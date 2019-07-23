from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

import numpy as np
import mglearn
import matplotlib.pyplot as plt

mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()


X, y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)

print("테스트 세트 예측:\n", reg.predict(X_test))
print("테스트 세트 R^2:{}:.2f".format(reg.score(X_test, y_test)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)

plt.rc("font", family="NanumGothic")

for n_neighborrs, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighborrs)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

    ax.set_title("{} 이웃의 훈련 스코어: {:.2f} 테스트 스코어 : {:2f}".format(n_neighborrs, reg.score(X_train, y_train), reg.score(X_test, y_test)), fontdict={"family":"NanumGothic"})
    ax.set_xlabel("특성", fontdict={"family":"NanumGothic"})
    ax.set_ylabel("타깃", fontdict={"family":"NanumGothic"})
axes[0].legend(["모델 예측", "훈련 데이터/타깃", "테스트 데이터/타깃"], loc="best")
plt.show()