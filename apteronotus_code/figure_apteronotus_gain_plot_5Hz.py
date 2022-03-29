import matplotlib.pyplot as plt
import numpy as np
import pylab
from IPython import embed
from scipy.optimize import curve_fit
from matplotlib.mlab import specgram
import os
from jar_functions import gain_curve_fit

plt.rcParams.update({'font.size': 12})

identifier = [#'2018lepto1',
              #'2018lepto4',
              #'2018lepto5',
              #'2018lepto76',
              '2018lepto98',
              #'2019lepto03',
              #'2019lepto24',
              #'2019lepto27',
              #'2019lepto30',
              #'2020lepto04',
              #'2020lepto06',
              '2020lepto16',
              '2020lepto19',
              '2020lepto20'
              ]

amfs = []
gains = []
taus = []
f_cs = []
predicts = []
for ID in identifier:
    predict = []

    print(ID)
    amf = np.load('5Hz_amf_%s.npy' %ID)
    amfs.append(amf)
    gain = np.load('5Hz_gain_%s.npy' %ID)
    gains.append(gain)

    sinv, sinc = curve_fit(gain_curve_fit, amf, gain, [2, 3])
    #print('tau:', sinv[0])
    taus.append(sinv[0])
    f_cutoff = abs(1 / (2*np.pi*sinv[0]))
    print('f_cutoff:', f_cutoff)
    f_cs.append(f_cutoff)

    # predict of gain
    for f in amf:
        G = np.max(gain) / np.sqrt(1 + (2 * ((np.pi * f * sinv[0]) ** 2)))
        predict.append(G)
    predicts.append(predict)

sort = sorted(zip(f_cs, identifier))
print(sort)
# order of plotting: 2018lepto98, 2020lepto16, 2020lepto19, 2020lepto19, 2020lepto20
# order of f_c: 2020lepto20, 2020lepto16, 2018lepto98, 2020lepto19

fig = plt.figure(figsize=(8.27,11.69))
ax0 = fig.add_subplot(221)
fig.text(0.05, 0.5, 'gain [Hz/(mV/cm)]', ha='center', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'envelope frequency [Hz]', ha='center', va='center')

ax0.set_xlim(0.0007, 1.5)
ax0.set_ylim(0.001, 10)
ax0.plot(amfs[0], gains[0],'o' , label = 'gain')
ax0.plot(amfs[0], predicts[0], label = 'fit')
ax0.axvline(x=f_cs[0], ymin=0, ymax=5, ls='-', alpha=0.5, label = 'cutoff frequency')
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.axes.xaxis.set_ticklabels([])
print('max[0]:', np.max(gain[0]))

ax1 = fig.add_subplot(222)
ax1.set_xlim(0.0007, 1.5)
ax1.set_ylim(0.001, 10)
ax1.plot(amfs[1], gains[1],'o' , label = 'gain')
ax1.plot(amfs[1], predicts[1], label = 'fit')
ax1.axvline(x=f_cs[1], ymin=0, ymax=5, ls='-', alpha=0.5, label = 'cutoff frequency')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.axes.yaxis.set_ticklabels([])
ax1.axes.xaxis.set_ticklabels([])
print('max[1]:', np.max(gain[1]))

ax2 = fig.add_subplot(223)
ax2.set_xlim(0.0007, 1.5)
ax2.set_ylim(0.001, 10)
ax2.plot(amfs[0], gains[0],'o' , label = 'gain')
ax2.plot(amfs[0], predicts[0], label = 'fit')
ax2.axvline(x=f_cs[0], ymin=0, ymax=5, ls='-', alpha=0.5, label = 'cutoff frequency')
ax2.set_xscale('log')
ax2.set_yscale('log')
print('max[2]:', np.max(gain[2]))

ax3 = fig.add_subplot(224)
ax3.set_xlim(0.0007, 1.5)
ax3.set_ylim(0.001, 10)
ax3.plot(amfs[2], gains[2],'o' , label = 'gain')
ax3.plot(amfs[2], predicts[2], label = 'fit')
ax3.axvline(x=f_cs[2], ymin=0, ymax=5, ls='-', alpha=0.5, label = 'cutoff frequency')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.axes.yaxis.set_ticklabels([])
print('max[3]:', np.max(gain[3]))

#plt.legend(loc = 'lower left')

plt.show()


#np.save('f_c', f_c)
#np.save('tau', tau)