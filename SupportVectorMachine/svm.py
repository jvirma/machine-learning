import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target


# print("Features: ", cancer.feature_names)
# print("Labels: ", cancer.target_names)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant', 'benign']

# clf = svm.SVC(kernel="poly", C=2)
clf = KNeighborsClassifier(n_neighbors=13)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)

print(acc)
