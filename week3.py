# %%
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
bostonHouseFrame = pd.read_csv("housing.data.csv", delimiter="\s+")
print(bostonHouseFrame)

# %%

with pd.option_context('mode.chained_assignment', None):
    bostonHouseTrainFrame, bostonHouseTestFrame = train_test_split(
        bostonHouseFrame, test_size=0.2, shuffle=True)
# %%
print('Number of instances in the original dataset is {}. After spliting Train has {} instances and test has {} instances.'.format(
    bostonHouseFrame.shape[0], bostonHouseTrainFrame.shape[0],  bostonHouseTestFrame.shape[0]))

# %%
# print(bostonHouseFrame)
# house_uniRM_x = bostonHouseFrame.loc[:,'RM']
#house_uniRM_x = house_uniRM_x.values.reshape(-1,1)
house_uniRM_x = bostonHouseFrame[['RM']]
house_y = bostonHouseFrame['MEDV']


# %%
print(house_y)
print(house_uniRM_x.shape)

# %%
trainX, testX, trainY, testY = train_test_split(
    house_uniRM_x, house_y, test_size=0.2, shuffle=True)
print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)
# %%
linReg = linear_model.LinearRegression()
linReg.fit(trainX, trainY)
# %%
# y-intercept or theta
print(linReg.intercept_)

# slope of univariate linear regression line or theta one
print(linReg.coef_)
# %%
pred_uniRM_y = linReg.predict(testX)
# %%
print('Mean squared error', mean_squared_error(testY, pred_uniRM_y))

# %%
plt.scatter(testX, testY, color='black')
plt.plot(testX, pred_uniRM_y, color='b', linewidth=3)

# %%
plt.scatter(testX, linReg.predict(testX) - testY, c='g', s=40)
