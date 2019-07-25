from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))


ridge = Ridge().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("훈련 세트 점수(A10): {:.2f}".format(ridge10.score(X_train, y_train)))
print("테스트 세트 점수(A10): {:.2f}".format(ridge10.score(X_test, y_test)))

ridge8 = Ridge(alpha=8).fit(X_train, y_train)
print("훈련 세트 점수(A8): {:.2f}".format(ridge8.score(X_train, y_train)))
print("테스트 세트 점수(A8): {:.2f}".format(ridge8.score(X_test, y_test)))

ridge6 = Ridge(alpha=6).fit(X_train, y_train)
print("훈련 세트 점수(A6): {:.2f}".format(ridge6.score(X_train, y_train)))
print("테스트 세트 점수(A6): {:.2f}".format(ridge6.score(X_test, y_test)))

ridge4 = Ridge(alpha=4).fit(X_train, y_train)
print("훈련 세트 점수(A4): {:.2f}".format(ridge4.score(X_train, y_train)))
print("테스트 세트 점수(A4): {:.2f}".format(ridge4.score(X_test, y_test)))

ridge2 = Ridge(alpha=2).fit(X_train, y_train)
print("훈련 세트 점수(A2): {:.2f}".format(ridge2.score(X_train, y_train)))
print("테스트 세트 점수(A2): {:.2f}".format(ridge2.score(X_test, y_test)))

ridge1 = Ridge(alpha=1).fit(X_train, y_train)
print("훈련 세트 점수(A1): {:.2f}".format(ridge1.score(X_train, y_train)))
print("테스트 세트 점수(A1): {:.2f}".format(ridge1.score(X_test, y_test)))

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("훈련 세트 점수(A0.1): {:.2f}".format(ridge01.score(X_train, y_train)))
print("테스트 세트 점수(A0.1): {:.2f}".format(ridge01.score(X_test, y_test)))


plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge1.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")

plt.xlabel("계수 목록", fontdict={"family":"NanumGothic"})
plt.ylabel("계수 크기", fontdict={"family":"NanumGothic"})
xlims = plt.xlim()

plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-25, 25)
plt.legend()

plt.show()

mglearn.plots.plot_ridge_n_samples()

