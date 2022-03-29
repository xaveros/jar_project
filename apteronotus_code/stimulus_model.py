from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from jar_functions import sin_response

plt.rcParams.update({'font.size': 27})

# AM model
lower = 0
upper = 200
sample = 1000
x = np.linspace(lower, upper, sample)
y1 = (sin_response(np.linspace(lower, upper, sample), 0.02, -np.pi/2, -0.75) - 1)
y2 = (sin_response(np.linspace(lower, upper, sample), 0.02, -np.pi/2, 0.75) + 1)

fig = plt.figure(figsize = (6,6))
# ax = fig.add_subplot(121)
# ax.plot(x, y1, c = 'red')
# ax.plot(x, y2, c = 'red')
# ax.fill_between(x, y1, y2)
#
# ax.set_xlabel('time[s]')
# ax.set_ylabel('amplitude')
# ax.set_xlim(0,200)
# ax.axes.yaxis.set_ticks([])
# plt.text(-0.1, 1.05, "A)", fontweight=550, transform=ax.transAxes)

# carrier
lower = 0
upper = 100
sample = 10000
x = np.linspace(lower, upper, sample)
y1 = (sin_response(np.linspace(lower, upper, sample), 800, np.pi, -0.75) - 1)

ax1 = fig.add_subplot(111)
ax1.plot(x, y1, lw = 4)
ax1.axhline(y = -0.25, c = 'red', lw = 4)
ax1.axhline(y = -1.75, c = 'red', lw = 4)

ax1.set_xlabel('time[ms]')
ax1.set_xlim(0,100)
ax1.axes.get_yaxis().set_visible(False)
plt.xticks((0,50,100), [0,5,10])
plt.show()