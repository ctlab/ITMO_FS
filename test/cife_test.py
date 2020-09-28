import numpy as np
import pandas as pd
from ITMO_FS.ensembles.cife import CIFE

cife = CIFE()
filename = 'datasets/diabetes.csv'
dataset = pd.read_csv(filename)
result = cife.fit(dataset.to_numpy())
print(result)