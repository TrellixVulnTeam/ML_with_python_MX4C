from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris_dataset = load_iris()

knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit()