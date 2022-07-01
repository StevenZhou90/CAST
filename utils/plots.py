import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

x = np.random.normal(0,1,(2,1000))
x = np.random.rand(1000,2)
x[0:500,0] = 0
x[500:1000,0] = 1
print(x.shape)
df = pd.DataFrame(x, columns=['one', 'two'])
print(df)


sns.displot(df, kde=True, x='two', col='one')
plt.show()
