import matplotlib.pyplot as plt
import numpy as np

a = np.linspace(0, 10, 11)
b = a**4
x = np.arange(0,10)
y = 2*x

# print(x)
fig = plt.figure(figsize=(8, 6), dpi=100)

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes1.plot(a, b)

axes2 = fig.add_axes([0.1, 0.2, 0.4, 0.4])
axes2.plot(x, y)
fig.savefig('./image/fig.png')
plt.show()
