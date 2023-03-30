# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# %%
bostonHouseFrame = pd.read_csv("housing.data.csv", delimiter="\s+")
print(bostonHouseFrame)

# %%
with pd.option_context('mode.chained_assignment', None):
    bostonHouseTrainFrame, bostonHouseTestFrame = train_test_split(
        bostonHouseFrame, test_size=0.2, shuffle=True)
# %%
# set up DV and IV
house_uniRM_x = bostonHouseFrame[['RM']]
house_y = bostonHouseFrame['MEDV']

# %%
trainX, testX, trainY, testY = train_test_split(
    house_uniRM_x, house_y, test_size=0.2, shuffle=True)
# print(trainX.shape)
# print(testX.shape)
# print(trainY.shape)
# print(testY.shape)
# %%
linReg = linear_model.LinearRegression()
# linReg.fit(house_uniRM_x, house_y)
polyReg = PolynomialFeatures(degree=4, include_bias=False)
polyFeatures = polyReg.fit_transform(trainX)
poly_reg_model = linear_model.LinearRegression()
# %%
linReg.fit(trainX, trainY)
poly_reg_model.fit(polyFeatures, trainY)

# %%
# calculate the predict value
pred_uniRM_y = linReg.predict(testX)
polyFeatures_test = polyReg.fit_transform(testX)
pred_uniRM_y_poly = poly_reg_model.predict(polyFeatures_test)

# %%
print(pred_uniRM_y)
print(pred_uniRM_y_poly)

# %%
plt.scatter(testX, testY, color='black')
plt.plot(testX, pred_uniRM_y, color='red')

orders = np.argsort(testX.ravel())
plt.plot(testX[orders], pred_uniRM_y_poly[orders], color='blue', linewidth=3)
plt.show()
# %%
