import numpy as np
import pandas as pd

from filters.VDM import *

data = pd.read_csv('data/speeddating.csv', low_memory=False)
complete_data = data.query('has_null == 0').iloc[:800, 1:]
filtered_data = complete_data.select_dtypes(include=['int', 'int32', 'int64'])
y = filtered_data.match
x = filtered_data.drop('match', axis=1)


xx = np.array([[0, 0, 0, 0],
               [1, 0, 1, 1],
               [1, 0, 0, 2]])
yy = np.array([0,
               1,
               1])

vdm = VDM()
s = vdm.run(xx, yy)

print(s)
