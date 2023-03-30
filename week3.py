# %%
from sklearn.model_selection import train_test_split
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
house_uniRM_x = bostonHouseFrame.loc[:'RM']
#house_uniRM_x = house_uniRM_x.values.reshape(-1,1)
house_uniRM_x = bostonHouseFrame[['RM']]
house_y = bostonHouseFrame['MEDV']
print(house_uniRM_x)
