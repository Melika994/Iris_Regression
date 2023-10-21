import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import metrics


iris = datasets.load_iris()

# convert to dataframe
df = pd.DataFrame(iris.data,columns=iris.feature_names)

df.shape
df.info()   # to find type of each feature
df.isnull().sum().sum()  # to find NaN data
df.describe()   # to find total information of data

# finding correlation
CORR = df.corr()   # to find the most correlated features

sns.heatmap(CORR, annot=True)
plt.show()

X_Train, X_Test, Y_Train, Y_Test = model_selection.train_test_split(df[["petal width (cm)"]],df[["petal length (cm)"]],
                                                                    train_size=0.8, random_state=5)
regr = linear_model.LinearRegression()
regr.fit(X_Train,Y_Train)

plt.scatter(df[["petal width (cm)"]],df[["petal length (cm)"]],  color='blue')
plt.plot(X_Train, regr.coef_[0][0]*X_Train + regr.intercept_[0], '-r')
plt.xlabel("petal width (cm)")
plt.ylabel("petal length (cm)")

Y_Test_ = regr.predict(X_Test)

print("Linear regression by considering single feature")
print("Mean absolute error: %.2f" % np.mean(np.absolute(Y_Test - Y_Test_)))
print("Residual sum of squares (MSE): %.2f" % np.mean((Y_Test - Y_Test_) ** 2))
print("R2-score: %.2f" % metrics.r2_score(Y_Test,Y_Test_) )


# -------------------Multiple Linear Regression----------------------------

from sklearn import model_selection

X_Train2, X_Test2, Y_Train2, Y_Test2 = model_selection.train_test_split(
                                        df[["petal width (cm)","sepal length (cm)"]],df[["petal length (cm)"]],
                                        test_size=0.8, random_state=5)

regr2 = linear_model.LinearRegression()
regr2.fit(X_Train2,Y_Train2)
Y_Test2_ = regr2.predict(X_Test2)

print("Linear regression by considering Multiple feature")
print("Mean absolute error: %.2f" % np.mean(np.absolute(Y_Test2 - Y_Test2_)))
print("Residual sum of squares (MSE): %.2f" % np.mean((Y_Test2 - Y_Test2_) ** 2))
print("R2-score: %.2f" % metrics.r2_score(Y_Test2,Y_Test2_))
