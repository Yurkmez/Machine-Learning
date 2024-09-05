import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,11, 10)

fig = plt.figure()

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

ax.plot(x, x, label='X vs X', color='red', ls='-.', marker= 'x', markersize=20)
# ax.plot(x, x**2, label='X vs X**2', color='green')
ax.plot(x, x**2, label='X vs X**2', color='#1c03fc', lw=5)
# или hex color picker
# ax.legend(loc='upper left')
# или
ax.legend(loc=(0.2, 0.5))


plt.title('Graph')
plt.xlabel('x')
plt.ylabel('y')

# plt.savefig('./image/1.png')
plt.show()