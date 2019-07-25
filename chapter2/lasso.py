from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split

import mglearn
import numpy as np
import matplotlib.pyplot as plt


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lasso = Lasso().fit(X_train, y_train)

print("훈련 세트 점수: {:.2f}".format(lasso.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso.score(X_test, y_test)))
print("사용한 특성의 개수:", np.sum(lasso.coef_ != 0))

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("훈련 세트 점수(A:0.01): {:.2f}".format(lasso001.score(X_train, y_train)))
print("테스트 세트 점수(A:0.01): {:.2f}".format(lasso001.score(X_test, y_test)))
print("사용한 특성의 개수(A:0.01): ", np.sum(lasso001.coef_ != 0))

lasso0001 = Lasso(alpha=0.001, max_iter=100000).fit(X_train, y_train)
print("훈련 세트 점수(A:0.001): {:.2f}".format(lasso0001.score(X_train, y_train)))
print("테스트 세트 점수(A:0.001): {:.2f}".format(lasso0001.score(X_test, y_test)))
print("사용한 특성의 개수(A:0.001): ", np.sum(lasso0001.coef_ != 0))

ridge01 = Ridge(alpha=0.01).fit(X_train, y_train)


plt.plot(lasso.coef_, "s", label="Lasso alpha=1")
plt.plot(lasso001.coef_, "^", label="Lasso alpha=0.01")
plt.plot(lasso0001.coef_, "v", label="Lasso alpha=0.001")
plt.plot(ridge01.coef_, "o", label="Ridge alpha=0,01")

plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("계수 목록", fontdict={"family": "NanumGothic"})
plt.xlabel("계수 크기", fontdict={"family": "NanumGothic"})
plt.show()