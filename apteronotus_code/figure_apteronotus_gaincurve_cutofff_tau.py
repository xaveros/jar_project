import matplotlib.pyplot as plt
import numpy as np
import pylab
from IPython import embed
from scipy.optimize import curve_fit
from matplotlib.mlab import specgram
import os
from jar_functions import gain_curve_fit

plt.rcParams.update({'font.size': 12})

identifier = ['2018lepto4']

tau = []
f_c = []
for ID in identifier:
    predict = []

    print(ID)
    amf = np.load('amf_%s.npy' %ID)
    gain = np.load('gain_%s.npy' %ID)
    print(gain)

    sinv, sinc = curve_fit(gain_curve_fit, amf, gain, [2, 3])
    print('tau:', sinv[0])
    tau.append(sinv[0])
    f_cutoff = abs(1 / (2*np.pi*sinv[0]))
    print('f_cutoff:', f_cutoff)
    f_c.append(f_cutoff)

    # predict of gain
    for f in amf:
        G = np.max(gain) / np.sqrt(1 + (2 * ((np.pi * f * sinv[0]) ** 2)))
        predict.append(G)
    print(np.max(gain))

    fig = plt.figure(figsize=(8.27, 11.69/2))
    ax = fig.add_subplot()
    ax.plot(amf, gain,'o' , label = 'gain')
    ax.plot(amf, predict, label = 'fit')
    ax.axvline(x=f_cutoff, ymin=0, ymax=5, ls='-', alpha=0.5, label = 'cutoff frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('gain [Hz/(mV/cm)]')
    ax.set_xlabel('envelope frequency [Hz]')
    ax.set_title('gaincurve %s' %ID)
    #plt.legend(loc = 'lower left')
    plt.show()


#np.save('f_c', f_c)
#np.save('tau', tau)