import numpy as np
from sklearn.linear_model import LinearRegression

from wrappers import RandomWrapper

fil = RandomWrapper(LinearRegression(), 8)
print(fil.fit(np.ones((5, 10)), np.ones(10)))
print(fil.predict(np.ones((5, 5))))
