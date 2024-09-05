import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,10)
y = 2*x

plt.plot(x, y)
plt.title('Graph')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 5)
plt.savefig('./image/1.png')
plt.show()