import numpy as np
import random
a = np.array([8, 9, 10, 11])
b = np.array([1, 2, 3, 4, 5, 6, 7])
b[:] = a[:3]
print(b)