import matplotlib.pyplot as plt
import numpy as np
import pylab
from IPython import embed
from scipy.optimize import curve_fit
from jar_functions import gain_curve_fit
from jar_functions import avgNestedLists
import matplotlib as mpl
from matplotlib import cm
import math

#plt.rcParams.update({'font.size': 16})

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

fig = plt.figure(figsize=(8.27, 11.69/2))
ax = fig.add_subplot(111)

custom_amf = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
tau = []
IDs = []
f_c = []
fit = []
fit_amf = []

all_gains = []
for ID in identifier:

    print(ID)
    IDs.append(ID)
    gain_10 = np.zeros(10)
    amf = np.load('5Hz_amf_%s.npy' % ID)
    gain = np.load('5Hz_gain_%s.npy' % ID)
    b = 0
    for aa, a in enumerate(custom_amf):
        if a in amf:
            gain_10[aa] = gain[b]
            b += 1
        else:
            gain_10[aa] = None
    print(gain_10)
    #print(amf)
    all_gains.append(gain_10)


    sinv, sinc = curve_fit(gain_curve_fit, amf, gain)
    print('tau:', sinv[0])
    tau.append(sinv[0])
    f_cutoff = abs(1 / (2*np.pi*sinv[0]))
    print('f_cutoff:', f_cutoff)
    f_c.append(f_cutoff)
    fit.append(gain_curve_fit(amf, *sinv))
    fit_amf.append(amf)
    ax.axvline(x=f_cutoff, ymin=0, ymax=5, color='C0', ls='-', alpha=0.5)

f_c_ID = zip(ID, f_c)

mean = []

g0 = []
g1 = []
g2 = []
g3 = []
g4 = []
g5 = []
g6 = []
g7 = []
g8 = []
g9 = []
for g in all_gains:
    if math.isnan(g[0]) is False:
        g0.append(g[0])
    if math.isnan(g[1]) is False:
        g1.append(g[1])
    if math.isnan(g[2]) is False:
        g2.append(g[2])
    if math.isnan(g[3]) is False:
        g3.append(g[3])
    if math.isnan(g[4]) is False:
        g4.append(g[4])
    if math.isnan(g[5]) is False:
        g5.append(g[5])
    if math.isnan(g[6]) is False:
        g6.append(g[6])
    if math.isnan(g[7]) is False:
        g7.append(g[7])
    if math.isnan(g[8]) is False:
        g8.append(g[8])
    if math.isnan(g[9]) is False:
        g9.append(g[9])
print(g0)
print(np.mean(g0))
print(g1)
print(np.mean(g1))
print(g2)
print(np.mean(g2))
print(g3)
print(np.mean(g3))
print(g4)
print(np.mean(g4))
print(g5)
print(np.mean(g5))
print(g6)
print(np.mean(g6))
print(g7)
print(np.mean(g7))
print(g8)
print(np.mean(g8))
print(g9)
print(np.mean(g9))

mean.append(np.mean(g0))
mean.append(np.mean(g1))
mean.append(np.mean(g2))
mean.append(np.mean(g3))
mean.append(np.mean(g4))
mean.append(np.mean(g5))
mean.append(np.mean(g6))
mean.append(np.mean(g7))
mean.append(np.mean(g8))
mean.append(np.mean(g9))

print('maximum of mean:', np.max(mean))

ax.plot(custom_amf, mean, 'o')

# uniformed: 2018lepto1, 2018lepto5,  2018lepto76,  2018lepto98, 2020lepto06, 2019lepto24, 2020lepto06
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('gain [Hz/(mV/cm)]')
ax.set_xlabel('envelope frequency [Hz]')
ax.set_xlim(0.0007, 1.5)
ax.set_ylim(0.001, 10)
plt.show()
embed()
