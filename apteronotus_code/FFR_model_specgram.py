import matplotlib.pyplot as plt
import numpy as np
import pylab
from IPython import embed
from scipy.optimize import curve_fit
from matplotlib.mlab import specgram
import os
from jar_functions import import_data
from jar_functions import import_amfreq
from jar_functions import sin_response

plt.rcParams.update({'font.size': 20})

base_path = 'D:\\jar_project\\JAR\\sin'

identifier = ['2018lepto98']
'''
specs = []
jars = []
sub_times = []
sub_lim0 = []
sub_lim1 = []
time = []

for ID in identifier:
    for dataset in os.listdir(os.path.join(base_path, ID)):
        if dataset == 'prerecordings':
            continue
        datapath = os.path.join(base_path, ID, dataset, '%s.nix' % dataset)
        print(datapath)

        amfreq = import_amfreq(datapath)
        if amfreq == '0.005' or amfreq == '0.02' or amfreq == '0.05':
            print(amfreq)

            data, pre_data, dt = import_data(datapath)

            #hstack concatenate: 'glue' pre_data and data
            if len(data) == 2:
                trace0 = np.hstack((pre_data[0], data[0]))
                trace1 = np.hstack((pre_data[1], data[1]))
            else:
                trace0 = np.hstack((pre_data, data))

            # data
            nfft = 2**17
            spec, freqs, times = specgram(trace0, Fs=1 / dt, detrend='mean', NFFT=nfft, noverlap=nfft * 0.95)
            dbspec = 10.0 * np.log10(spec)  # in dB
            power = dbspec[:, 25]

            fish_p = power[(freqs > 200) & (freqs < 1000)]
            fish_f = freqs[(freqs > 200) & (freqs < 1000)]

            index = np.argmax(fish_p)
            eodf = fish_f[index]
            eodf4 = eodf * 4

            lim0 = eodf4 - 10
            lim1 = eodf4 + 25

            df = freqs[1] - freqs[0]
            ix0 = int(np.floor(lim0/df))    # back to index
            ix1 = int(np.ceil(lim1/df))    # back to index
            spec4= dbspec[ix0:ix1, :]
            freq4 = freqs[ix0:ix1]
            jar4 = freq4[np.argmax(spec4, axis=0)]      # all freqs at max specs over axis 0

            cut_time_jar = times[:len(jar4)]
            specs.append(spec4)
            jars.append(jar4)
            sub_times.append(cut_time_jar)
            sub_lim0.append(lim0)
            sub_lim1.append(lim1)
            time.append(times)

np.save('spec0.npy', specs[0])
np.save('spec1.npy', specs[1])
np.save('spec2.npy', specs[2])
np.save('jar0.npy', jars[0])
np.save('jar1.npy', jars[1])
np.save('jar2.npy', jars[2])
np.save('sub_times0.npy', sub_times[0])
np.save('sub_times1.npy', sub_times[1])
np.save('sub_times2.npy', sub_times[2])
np.save('sub_lim0_0.npy', sub_lim0[0])
np.save('sub_lim0_1.npy', sub_lim0[1])
np.save('sub_lim0_2.npy', sub_lim0[2])
np.save('sub_lim1_0.npy', sub_lim1[0])
np.save('sub_lim1_1.npy', sub_lim1[1])
np.save('sub_lim1_2.npy', sub_lim1[2])
np.save('time0.npy', time[0])
np.save('time1.npy', time[1])
np.save('time2.npy', time[2])
'''
spec0 = np.load('spec0.npy')
spec1 = np.load('spec1.npy')
spec2 = np.load('spec2.npy')
jar0 = np.load('jar0.npy')
jar1 = np.load('jar1.npy')
jar2 = np.load('jar2.npy')
sub_times0 = np.load('sub_times0.npy')
sub_times1 = np.load('sub_times1.npy')
sub_times2 = np.load('sub_times2.npy')
sub_lim0_0 = np.load('sub_lim0_0.npy')
sub_lim0_1 = np.load('sub_lim0_1.npy')
sub_lim0_2 = np.load('sub_lim0_2.npy')
sub_lim1_0 = np.load('sub_lim1_0.npy')
sub_lim1_1 = np.load('sub_lim1_1.npy')
sub_lim1_2 = np.load('sub_lim1_2.npy')
time0 = np.load('time0.npy')
time1 = np.load('time1.npy')
time2 = np.load('time2.npy')

fig = plt.figure(figsize = (20,20))
ax0 = fig.add_subplot(232)
ax0.tick_params(width = 2, length = 5)
ax0.imshow(spec0, cmap='jet', origin='lower', extent=(time0[0], time0[-1], sub_lim0_0, sub_lim1_0), aspect='auto', vmin=-80, vmax=-10)
#ax0.plot(sub_times0, jar0, 'k', label = 'peak detection trace', lw = 2)
ax0.set_xlim(time0[0],time0[-1])
ax0.axes.xaxis.set_ticklabels([])
ax0.axes.yaxis.set_ticklabels([])

ax1 = fig.add_subplot(231)
ax1.tick_params(width = 2, length = 5)
ax1.imshow(spec1, cmap='jet', origin='lower', extent=(time1[0], time1[-1], sub_lim0_1, sub_lim1_1), aspect='auto', vmin=-80, vmax=-10)
#ax1.plot(sub_times1, jar1, 'k', label = 'peak detection trace', lw = 2)
ax1.set_xlim(time1[0],time1[-1])
ax1.set_ylabel('frequency [Hz]')
ax1.axes.xaxis.set_ticklabels([])

plt.text(-0.1, 1.05, "A)", fontweight=550, transform=ax1.transAxes)

ax2 = fig.add_subplot(233)
ax2.tick_params(width = 2, length = 5)
ax2.imshow(spec2, cmap='jet', origin='lower', extent=(time2[0], time2[-1], sub_lim0_2, sub_lim1_2), aspect='auto', vmin=-80, vmax=-10)
#ax2.plot(sub_times2, jar2, 'k', label = 'peak detection trace', lw = 2)
ax2.set_xlim(time2[0],time2[-1])
ax2.axes.xaxis.set_ticklabels([])
ax2.axes.yaxis.set_ticklabels([])

# AM model: 0.05 Hz

lower0 = 50
upper0 = 250
sample0 = 2000
x0 = np.linspace(lower0, upper0, sample0)
y0_0 = (sin_response(np.linspace(lower0, upper0, sample0), 0.05, np.pi/2, -0.35) - 0.5)
y0_1 = (sin_response(np.linspace(lower0, upper0, sample0), 0.05, np.pi/2, 0.35) + 0.5)

ax3 = fig.add_subplot(234)
ax3.tick_params(width = 2, length = 5)
plt.hlines(y = 0, xmin = 0, xmax = 50,  color = 'red')
plt.vlines(x = 50, ymin = -0.15, ymax = 0.15,  color = 'red')
ax3.plot(x0, y0_0, c = 'red')
ax3.plot(x0, y0_1, c = 'red')
ax3.fill_between(x0, y0_0, y0_1)

ax3.set_ylabel('amplitude [mV/cm]')
ax3.set_xlabel('time [s]')
ax3.set_xlim(0,250)

plt.text(-0.1, 1.05, "B)", fontweight=550, transform=ax3.transAxes)

# AM model: 0.02 Hz
lower1 = 50
upper1 = 250
sample1 = 2000
x1 = np.linspace(lower1, upper1, sample1)
y1_0 = (sin_response(np.linspace(lower1, upper1, sample1), 0.02, -np.pi/2 , -0.35) - 0.5)
y1_1 = (sin_response(np.linspace(lower1, upper1, sample1), 0.02, -np.pi/2, 0.35) + 0.5)

ax4 = fig.add_subplot(235)
ax4.tick_params(width = 2, length = 5)
plt.hlines(y = 0, xmin = 0, xmax = 50,  color = 'red')
plt.vlines(x = 50, ymin = -0.15, ymax = 0.15,  color = 'red')
ax4.plot(x1, y1_0, c = 'red')
ax4.plot(x1, y1_1, c = 'red')
ax4.fill_between(x1, y1_0, y1_1)

ax4.set_xlabel('time [s]')
ax4.set_xlim(0,250)
ax4.axes.yaxis.set_ticklabels([])

# AM model: 0.005 Hz
lower2 = 50
upper2 = 450
sample2 = 2000
x2 = np.linspace(lower2, upper2, sample2)
y2_0 = (sin_response(np.linspace(lower2, upper2, sample2), 0.005, -np.pi , -0.35) - 0.5)
y2_1 = (sin_response(np.linspace(lower2, upper2, sample2), 0.005, -np.pi, 0.35) + 0.5)

ax5 = fig.add_subplot(236)
ax5.tick_params(width = 2, length = 5)
plt.hlines(y = 0, xmin = 0, xmax = 50,  color = 'red')
plt.vlines(x = 50, ymin = -0.15, ymax = 0.15,  color = 'red')
ax5.plot(x2, y2_0, c = 'red')
ax5.plot(x2, y2_1, c = 'red')
ax5.fill_between(x2, y2_0, y2_1)

ax5.set_xlabel('time [s]')
ax5.set_xlim(0,450)
ax5.axes.yaxis.set_ticklabels([])

plt.show()
embed()
