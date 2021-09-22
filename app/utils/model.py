from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

np.random.seed(7)

b = np.random.randint(20)

x, y = [],[]
for i in range(50):
    x.append(i + np.random.normal(0,6,1) + b)
    y.append(i - np.random.normal(0,3,1) + b*2)

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

plt.plot(x, y)