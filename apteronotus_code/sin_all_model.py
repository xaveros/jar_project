import matplotlib.pyplot as plt
import numpy as np
import pylab
from IPython import embed
from scipy.optimize import curve_fit
from jar_functions import gain_curve_fit
from jar_functions import avgNestedLists
from matplotlib import gridspec

#plt.rcParams.update({'font.size': 16})

amf = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

low_lim = 0.005
high_lim = 6

# subplot 1
fig = plt.figure(figsize=(8.27,11.69))
gs = gridspec.GridSpec(2, 2)
ax0 = fig.add_subplot(gs[0,:])
fig.text(0.06, 0.5, 'gain [Hz/(mV/cm)]', ha='center', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'envelope frequency [Hz]', ha='center', va='center')

custom_f = np.logspace(-2, -1, 4)
custom_alpha = np.logspace(0.6, 0.1, 4)
c_gain = []
custom_tau = abs(1 / (2 * np.pi * custom_f))
for t, a in zip(custom_tau, custom_alpha):
    custom_gain = []
    for am in amf:
        custom_g = gain_curve_fit(am, t, a)
        custom_gain.append(custom_g)
    c_gain.append(custom_gain)
col = ['blue', 'orange', 'green', 'purple']
for cc, c in enumerate(c_gain):
    ax0.plot(amf, c, c = col[cc])
    ax0.axvline(x=custom_f[cc], c = col[cc], ymin=0, ymax=5, alpha=0.5)

mean = avgNestedLists(c_gain)

ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_ylim(low_lim, high_lim)
ax0.plot(amf, mean, lw = 3, c = 'r')

# subplot 2
ax1 = fig.add_subplot(gs[1,0])

custom_f = np.logspace(-2, -1, 10)
custom_alpha = np.logspace(0.6, 0.1, 10)
c_gain = []
custom_tau = abs(1 / (2 * np.pi * custom_f))
for t, a in zip(custom_tau, custom_alpha):
    custom_gain = []
    for am in amf:
        custom_g = gain_curve_fit(am, t, a)
        custom_gain.append(custom_g)
    c_gain.append(custom_gain)
col = ['blue', 'orange', 'green']
for cc, c in enumerate(c_gain):
    ax1.plot(amf, c, c = 'C0')
    ax1.axvline(x=custom_f[cc], c = 'C0', ymin=0, ymax=5, alpha=0.5)  # colors_uniform[ff])

mean = avgNestedLists(c_gain)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(low_lim, high_lim)
ax1.plot(amf, mean, lw = 3, c = 'r')

# subplot 3
ax2 = fig.add_subplot(gs[1,1])

custom_f = np.logspace(-2.75, -0.25, 10)
custom_alpha = np.logspace(0.6, 0.1, 10)
c_gain = []
custom_tau = abs(1 / (2 * np.pi * custom_f))
for t, a in zip(custom_tau, custom_alpha):
    custom_gain = []
    for am in amf:
        custom_g = gain_curve_fit(am, t, a)
        custom_gain.append(custom_g)
    c_gain.append(custom_gain)
col = ['blue', 'orange', 'green']
for cc, c in enumerate(c_gain):
    ax2.plot(amf, c, c = 'C0')
    ax2.axvline(x=custom_f[cc], c = 'C0', ymin=0, ymax=5, alpha=0.5)  # colors_uniform[ff])

mean = avgNestedLists(c_gain)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim(low_lim, high_lim)
ax2.set_yticklabels([])
ax2.plot(amf, mean, lw = 3, c = 'r')
plt.show()

embed()

