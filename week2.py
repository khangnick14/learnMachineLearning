# %%

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
list1 = [3, -4, 6, 'RMIT', 12]
print(list1[-1])


# %%
list1[1:10:1]


# %%
list1[-3::]

# %%
list1[2::]

# %%
list1[:3]

# %%
list1 = [3, -4, 6, 'RMIT', 12]
list1.append('RMITTTT')
print(list1)

# %%
list1 = [3, -4, 6, 'RMIT', 12]
list1.insert(2, "RMITTTT")
print(list1)

# %%
list1 = [3, -4, 6, 'RMIT', 12]
list1.remove(3)
print(list1)

# %%
list_temp = [32, 35, 25, 20, 31, 30, 29]
print('Temp > 30')
for temp in list_temp:
    if temp > 30:
        print(temp)

# %%
high_temp = []
for temp in list_temp:
    if temp > 30:
        high_temp.append(temp)

print(high_temp)

# %%
high_id = []
for temp in list_temp:
    if temp > 30:
        high_id.append(list_temp.index(temp))

print(high_id)

# %%

arr1 = np.array([32, 35, 31, 28])
print(arr1[-1])

# %%
data = pd.read_csv('housing.data.csv', delimiter='\s+')
data.info()

# %%
data.head()
# %%
data.describe()

# %%
# row 1st, column 2nd
data.iloc[0, 2]

# %%
data.iloc[[0, 1, 2], [0, 1]]

# %%
data.plot(kind='scatter', y='MEDV', x='CRIM')
data.show()

# %%
# (506, 14): show entries and columns
data.shape
# %%
data.info()

# %%
# compute the min of each column
pd.DataFrame.min(data)
# %%
pd.DataFrame.max(data)

# %%
pd.DataFrame.mean(data)
# %%
pd.DataFrame.median(data)
# %%
plt.figure(figsize=(20, 20), dpi=80)
data.hist()
plt.show()

# %%
plt.figure(figsize=(20, 20))
for i, col in enumerate(data.columns):
    plt.subplot(4, 5, i+1)
    plt.hist(data[col], alpha=0.3, color='b', density=True)
    plt.title(col)
    plt.xticks(rotation='vertical')

# %%
data.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False)
plt.show()
# %%
plt.boxplot(data['TAX'])
plt.title('Median House Price')
plt.show()
# %%
correlation = data.corr()
fig = plt.figure(figsize=(16, 16), dpi=80)
ax = fig.add_subplot(111)
cax = ax.matshow(correlation, vmin=-1, vmax=1, cmap=plt.cm.PuBu)
fig.colorbar(cax)
ticks = np.arange(0, 14, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()
# %%
# use seaborn library
f, ax = plt.subplots(figsize=(11, 9))
corr = data.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0,
                 cmap=sns.diverging_palette(20, 220, n=200), square=True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right'
)

# %%
