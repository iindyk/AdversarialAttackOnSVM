import numpy as np

h = [1, 2, 3, 4, 5, 6, 7 ,8]
h_np = np.reshape(h, (2, 4))
h_np = np.transpose(h_np)
print(h_np)