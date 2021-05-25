# -*- coding: utf-8 -*-
"""Seaborn-Prac.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pf6qtkxjVRLjrL1fA71Qr7927MTrVxhb
"""

# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
# %matplotlib inline

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100

tips = pd.read_csv('https://raw.githubusercontent.com/btkhimsar/DataSets/master/tips.csv')
tips.head()

sns.relplot(x='total_bill',y='tip', color='r', data=tips)

sns.relplot(x='total_bill', y='tip', hue='smoker', palette='viridis', data=tips)

sns.relplot(x='total_bill', y='tip', hue='smoker', style='time', data=tips, palette='viridis')

sns.relplot(x='total_bill',y='tip',size='size',sizes=(20,200),data=tips);

df = pd.DataFrame(dict(time=np.arange(500), value=np.random.randn(500).cumsum()))

g = sns.relplot(x='time', y='value', kind='line', data=df)

g.fig.autofmt_xdate()

df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])
sns.relplot(x='x', y='y', sort=False , kind='line', data=df);

fmri = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/fmri.csv')
fmri.head()

sns.relplot(x='timepoint', y='signal', kind='line', data=fmri, color = 'purple');

"""CONFIDENCE INTERVALS **ci** ARE TIME-INTENSIVE FOR LARGER DATASETS.
Disable them 


```
ci = None
```


"""

sns.relplot(x='timepoint', y='signal', ci=None, kind='line', color='blue', data=fmri)