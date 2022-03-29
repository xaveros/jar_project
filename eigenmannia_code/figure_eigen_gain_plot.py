import matplotlib.pyplot as plt
import numpy as np
import pylab
from IPython import embed
from scipy.optimize import curve_fit
from matplotlib.mlab import specgram
import os
from jar_functions import gain_curve_fit

plt.rcParams.update({'font.size': 10})

identifier = ['2015eigen8',
              '2015eigen15',
              '2015eigen16',
              '2015eigen17',
              '2015eigen19'
              ]

amfs = []
gains = []
maxgains = []
mingains = []
taus = []
f_cs = []
predicts = []
for ID in identifier:
    predict = []

    print(ID)
    amf = np.load('eigen_amf_%s.npy' % ID)
    amfs.append(amf)
    gain = np.load('eigen_gain_%s.npy' % ID)
    gains.append(gain)
    print(np.max(gain))
    sinv, sinc = curve_fit(gain_curve_fit, amf, gain, [2, 3])
    # print('tau:', sinv[0])
    taus.append(sinv[0])
    f_cutoff = abs(1 / (2 * np.pi * sinv[0]))
    print('f_cutoff:', f_cutoff)
    f_cs.append(f_cutoff)

    print('min gain:', np.min(gain))
    print('max gain:', np.max(gain))
    maxgains.append(np.max(gain))
    mingains.append(np.min(gain))
    # predict of gain
    for f in amf:
        G = np.max(gain) / np.sqrt(1 + (2 * ((np.pi * f * sinv[0]) ** 2)))
        predict.append(G)
    predicts.append(predict)

print('max of absolute max gain:', np.max(maxgains))
print('min of absolute max gain:', np.min(maxgains))
print('max of absolute min gain:', np.max(mingains))
print('min of absolute min gain:', np.min(mingains))
print('absolute max f_c:', np.max(f_cs))
print('absolute min f_c:', np.min(f_cs))

sort = sorted(zip(f_cs, identifier))
print(sort)
# order of plotting: 2018lepto1, 2018lepto5, 2018lepto76, 2018lepto98, 2019lepto24, 2020lepto06
# order of f_c: 2019lepto24, 2020lepto06, 2018lepto98, 2018lepto76, 2018lepto1, 2018lepto5

fig = plt.figure(figsize=(8.27, 11.69))
# ax0 = plt.subplot2grid(shape=(3,4), loc=(0,0), colspan = 2)
# ax1 = plt.subplot2grid((3,4), (0,2), colspan = 2)
# ax2 = plt.subplot2grid((3,4), (1,0), colspan = 2)
# ax3 = plt.subplot2grid((3,4), (1,2), colspan = 2)
# ax4 = plt.subplot2grid((3,4), (2,0), colspan = 2)

ax0 = fig.add_subplot(321)
fig.text(0.05, 0.5, 'gain [Hz/(mV/cm)]', ha='center', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'envelope frequency [Hz]', ha='center', va='center')

ax0.set_xlim(0.0007, 1.5)
ax0.set_ylim(0.001, 10)
ax0.plot(amfs[1], gains[1], 'o', label='gain')
ax0.plot(amfs[1], predicts[1], label='fit')
ax0.axvline(x=f_cs[1], ymin=0, ymax=5, ls='-', alpha=0.5, label='cutoff frequency')
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.axes.xaxis.set_ticklabels([])

ax1 = fig.add_subplot(322)
ax1.set_xlim(0.0007, 1.5)
ax1.set_ylim(0.001, 10)
ax1.plot(amfs[0], gains[0], 'o', label='gain')
ax1.plot(amfs[0], predicts[0], label='fit')
ax1.axvline(x=f_cs[0], ymin=0, ymax=5, ls='-', alpha=0.5, label='cutoff frequency')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.axes.yaxis.set_ticklabels([])
ax1.axes.xaxis.set_ticklabels([])

ax2 = fig.add_subplot(323)
ax2.set_xlim(0.0007, 1.5)
ax2.set_ylim(0.001, 10)
ax2.plot(amfs[4], gains[4], 'o', label='gain')
ax2.plot(amfs[4], predicts[4], label='fit')
ax2.axvline(x=f_cs[4], ymin=0, ymax=5, ls='-', alpha=0.5, label='cutoff frequency')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.axes.xaxis.set_ticklabels([])

ax3 = fig.add_subplot(324)
ax3.set_xlim(0.0007, 1.5)
ax3.set_ylim(0.001, 10)
ax3.plot(amfs[2], gains[2], 'o', label='gain')
ax3.plot(amfs[2], predicts[2], label='fit')
ax3.axvline(x=f_cs[2], ymin=0, ymax=5, ls='-', alpha=0.5, label='cutoff frequency')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.axes.yaxis.set_ticklabels([])

ax4 = fig.add_subplot(325)
ax4.set_xlim(0.0007, 1.5)
ax4.set_ylim(0.001, 10)
ax4.plot(amfs[3], gains[3], 'o', label='gain')
ax4.plot(amfs[3], predicts[3], label='fit')
ax4.axvline(x=f_cs[3], ymin=0, ymax=5, ls='-', alpha=0.5, label='cutoff frequency')
ax4.set_xscale('log')
ax4.set_yscale('log')

# plt.legend(loc = 'lower left')
plt.show()

# np.save('f_c', f_c)
# np.save('tau', tau)
