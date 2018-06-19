from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# load data
import xgboost

X = loadtxt('x2.csv', delimiter=",")
Y = loadtxt('y2.csv')

# split data into train and test sets
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=123)

# fit model no training data
model = xgboost.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123, max_depth=30)
print("before fit")
model.fit(X_train, y_train)
xgboost.plot_tree(model)
plt.show()
print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))