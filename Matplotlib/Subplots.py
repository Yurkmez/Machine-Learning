import matplotlib.pyplot as plt
import numpy as np

a = np.linspace(0, 10, 11)
b = a**4
x = np.arange(0,10)
y = 2*x

fig,axes = plt.subplots(nrows=2, ncols=2 )
axes[0][0].plot(x, y)
axes[0][0].set_xlabel('X')
axes[0][0].set_ylabel('Y')
axes[0][0].set_title('TITLE')

axes[1][1].plot(a, b)
axes[1][1].set_xlabel('A')
axes[1][1].set_ylabel('B')

fig.suptitle('Super TITLE')
# Чтобы шкалы осей не накладывались друг на друга
# plt.tight_layout()
# То же вручную
fig.subplots_adjust(wspace=0.4, hspace=0.4)
fig.savefig('./image/fig.png')
#  - чтобы вокруг графика ничего не отсекалось
fig.savefig('./image/fig_2.png', bbox_inches='tight')

plt.show()
