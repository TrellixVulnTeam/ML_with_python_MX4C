from matplotlib import pyplot as plt
import matplotlib
import mglearn


X, y = mglearn.datasets.make_forge()

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

## 한글........

font = {"family": "NanumGothic", "weight": "normal"}

plt.rc("font", family=font["family"])
plt.rc("axes", unicode_minus=False)

plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("첫 번째 특성", fontdict=font)
plt.ylabel("두 번째 특성", fontdict=font)
plt.title("테스트테스트", fontdict=font)
print("X.shape", X.shape)
plt.show()

X, y  = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3, 3)
plt.xlabel("특성", fontdict=font)
plt.ylabel("타깃", fontdict=font)
plt.show()
