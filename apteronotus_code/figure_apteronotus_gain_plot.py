import matplotlib.pyplot as plt
import numpy as np
import pylab
from IPython import embed
from scipy.optimize import curve_fit
from matplotlib.mlab import specgram
import os
from jar_functions import gain_curve_fit

plt.rcParams.update({'font.size': 10})

identifier = ['2018lepto1',
              #'2018lepto4',
              '2018lepto5',
              '2018lepto76',
              '2018lepto98',
              #'2019lepto03',
              '2019lepto24',
              #'2019lepto27',
              #'2019lepto30',
              #'2020lepto04',
              '2020lepto06',
              #'2020lepto16',
              #'2020lepto19',
              #'2020lepto20'
              ]

amfs = []
gains = []
taus = []
f_cs = []
predicts = []
maxgains = []
for ID in identifier:
    predict = []

    print(ID)
    amf = np.load('amf_%s.npy' %ID)
    amfs.append(amf)
    gain = np.load('gain_%s.npy' %ID)
    gains.append(gain)

    sinv, sinc = curve_fit(gain_curve_fit, amf, gain, [2, 3])
    #print('tau:', sinv[0])
    taus.append(sinv[0])
    f_cutoff = abs(1 / (2*np.pi*sinv[0]))
    print('f_cutoff:', f_cutoff)
    f_cs.append(f_cutoff)

    # predict of gain
    print('max gain:', np.max(gain))
    maxgains.append(np.max(gain))

    for f in amf:
        G = np.max(gain) / np.sqrt(1 + (2 * ((np.pi * f * sinv[0]) ** 2)))
        predict.append(G)
    predicts.append(predict)
print('absolute max gain:', np.max(maxgains))
print('absolute min gain:', np.min(maxgains))
print('absolute max f_c:', np.max(f_cs))
print('absolute min f_c:', np.min(f_cs))


sort = sorted(zip(f_cs, identifier))
print(sort)
# order of plotting: 2018lepto1, 2018lepto5, 2018lepto76, 2018lepto98, 2019lepto24, 2020lepto06
# order of f_c: 2019lepto24, 2020lepto06, 2018lepto98, 2018lepto76, 2018lepto1, 2018lepto5

fig = plt.figure(figsize=(8.27,11.69))
ax0 = fig.add_subplot(321)
fig.text(0.05, 0.5, 'gain [Hz/(mV/cm)]', ha='center', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'envelope frequency [Hz]', ha='center', va='center')

ax0.set_xlim(0.0007, 1.5)
ax0.set_ylim(0.001, 10)
ax0.plot(amfs[4], gains[4],'o' , label = 'gain')
ax0.plot(amfs[4], predicts[4], label = 'fit')
ax0.axvline(x=f_cs[4], ymin=0, ymax=5, ls='-', alpha=0.5, label = 'cutoff frequency')
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.axes.xaxis.set_ticklabels([])

ax1 = fig.add_subplot(322)
ax1.set_xlim(0.0007, 1.5)
ax1.set_ylim(0.001, 10)
ax1.plot(amfs[5], gains[5],'o' , label = 'gain')
ax1.plot(amfs[5], predicts[5], label = 'fit')
ax1.axvline(x=f_cs[5], ymin=0, ymax=5, ls='-', alpha=0.5, label = 'cutoff frequency')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.axes.yaxis.set_ticklabels([])
ax1.axes.xaxis.set_ticklabels([])

ax2 = fig.add_subplot(323)
ax2.set_xlim(0.0007, 1.5)
ax2.set_ylim(0.001, 10)
ax2.plot(amfs[3], gains[3],'o' , label = 'gain')
ax2.plot(amfs[3], predicts[3], label = 'fit')
ax2.axvline(x=f_cs[3], ymin=0, ymax=5, ls='-', alpha=0.5, label = 'cutoff frequency')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.axes.xaxis.set_ticklabels([])

ax3 = fig.add_subplot(324)
ax3.set_xlim(0.0007, 1.5)
ax3.set_ylim(0.001, 10)
ax3.plot(amfs[2], gains[2],'o' , label = 'gain')
ax3.plot(amfs[2], predicts[2], label = 'fit')
ax3.axvline(x=f_cs[2], ymin=0, ymax=5, ls='-', alpha=0.5, label = 'cutoff frequency')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.axes.yaxis.set_ticklabels([])
ax3.axes.xaxis.set_ticklabels([])

ax4 = fig.add_subplot(325)
ax4.set_xlim(0.0007, 1.5)
ax4.set_ylim(0.001, 10)
ax4.plot(amfs[0], gains[0],'o' , label = 'gain')
ax4.plot(amfs[0], predicts[0], label = 'fit')
ax4.axvline(x=f_cs[0], ymin=0, ymax=5, ls='-', alpha=0.5, label = 'cutoff frequency')
ax4.set_xscale('log')
ax4.set_yscale('log')

ax5 = fig.add_subplot(326)
ax5.set_xlim(0.0007, 1.5)
ax5.set_ylim(0.001, 10)
ax5.plot(amfs[1], gains[1],'o' , label = 'gain')
ax5.plot(amfs[1], predicts[1], label = 'fit')
ax5.axvline(x=f_cs[1], ymin=0, ymax=5, ls='-', alpha=0.5, label = 'cutoff frequency')
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.axes.yaxis.set_ticklabels([])

#plt.legend(loc = 'lower left')
plt.show()


#np.save('f_c', f_c)
#np.save('tau', tau)