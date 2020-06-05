import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
import pathlib
style.use("ggplot")


currentPath = str(pathlib.Path(__file__).parent.absolute())

data = pd.read_csv( currentPath + "\student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

#Training multiple times

# best = 0
# for _ in range(30):
#   x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
#   linear = linear_model.LinearRegression()
#   linear.fit(x_train, y_train)
#   acc = linear.score(x_test, y_test) # acc stands for accuracy
#   print(acc)
#   if acc > best:
#     best = acc
#     with open("studentgrades.pickle", "wb") as f:
#       pickle.dump(linear, f)

# Load pickle model

pickle_in = open(currentPath + "\studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)
print("-------------------------")
print('Coefficient: \n', linear.coef_) # These are each slope value
print('Intercept: \n', linear.intercept_) # This is the intercept
print("-------------------------")

predictions = linear.predict(x_test) # Gets a list of all predictions
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# Drawing and plotting model
plot = "G2"
pyplot.scatter(data[plot], data["G3"])
pyplot.legend(loc=4)
pyplot.xlabel(plot)
pyplot.ylabel("Final Grade")
pyplot.show()