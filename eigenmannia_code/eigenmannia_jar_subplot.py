import matplotlib.pyplot as plt
import numpy as np
import os
import nix_helpers as nh
from IPython import embed
from matplotlib.mlab import specgram
#from tqdm import tqdm
from jar_functions import parse_stimuli_dat
from jar_functions import norm_function_eigen
from jar_functions import mean_noise_cut_eigen
from jar_functions import get_time_zeros
from jar_functions import import_data_eigen
from scipy.signal import savgol_filter

plt.rcParams.update({'font.size': 10})

base_path = 'D:\\jar_project\\JAR\\eigenmannia\\deltaf'

#2015eigen8 no nix files
identifier = [#'2013eigen13',
              '2015eigen16'] #,'2015eigen17', '2015eigen19', '2020eigen22','2020eigen32']


response = []
deltaf = []

specs = []
jars = []
sub_times = []
sub_lim0 = []
sub_lim1 = []

for ID in identifier:
    for dataset in os.listdir(os.path.join(base_path, ID)):
        datapath = os.path.join(base_path, ID, dataset, '%s.nix' % dataset)
        #print(datapath)
        stimuli_dat = os.path.join(base_path, ID, dataset, 'manualjar-eod.dat')
        #print(stimuli_dat)
        delta_f, duration = parse_stimuli_dat(stimuli_dat)
        dur = int(duration[0][0:2])
        if delta_f == [-2.0] or delta_f == [2.0] or delta_f == [-10.0] or delta_f == [10.0]:
            print(delta_f)

            data, pre_data, dt = import_data_eigen(datapath)
            # hstack concatenate: 'glue' pre_data and data
            dat = np.hstack((pre_data, data))

            # data
            nfft = 2 ** 17
            spec, freqs, times = specgram(dat[0], Fs=1 / dt, detrend='mean', NFFT=nfft, noverlap=nfft * 0.95)
            dbspec = 10.0 * np.log10(spec)  # in dB
            power = dbspec[:, 25]

            fish_p = power[(freqs > 200) & (freqs < 1000)]
            fish_f = freqs[(freqs > 200) & (freqs < 1000)]

            index = np.argmax(fish_p)
            eodf = fish_f[index]
            eodf4 = eodf * 4

            lim0 = eodf4 - 42
            lim1 = eodf4 + 42

            df = freqs[1] - freqs[0]
            ix0 = int(np.floor(lim0 / df))  # back to index
            ix1 = int(np.ceil(lim1 / df))  # back to index
            spec4 = dbspec[ix0:ix1, :]
            freq4 = freqs[ix0:ix1]
            jar4 = freq4[np.argmax(spec4, axis=0)]  # all freqs at max specs over axis 0

            cut_time_jar = times[:len(jar4)]
            ID_delta_f = [ID, str(delta_f[0]).split('.')[0]]

            b = []
            for idx, i in enumerate(times):
                if i > 0 and i < 10:
                    b.append(jar4[idx])
            j = []
            for idx, i in enumerate(times):
                if i > 15 and i < 55:
                    j.append(jar4[idx])

            r = np.median(j) - np.median(b)
            print('response:', r)
            deltaf.append(delta_f[0])
            response.append(r)
            specs.append(spec4)
            jars.append(jar4)
            sub_times.append(cut_time_jar)
            sub_lim0.append(lim0)
            sub_lim1.append(lim1)
        if len(specs) == 4:
            break

    # plt.imshow(specs[0], cmap='jet', origin='lower', extent=(times[0], times[-1], sub_lim0[0], sub_lim1[1]), aspect='auto', vmin=-80, vmax=-10)
    # plt.plot(sub_times[0], jars[0], 'k', label = 'peak detection trace', lw = 2)
    # plt.hlines(y=lim0 + 5, xmin=10, xmax=70, lw=4, color='yellow', label='stimulus duration')
    # plt.hlines(y=lim0 + 5, xmin=0, xmax=10, lw=4, color='red', label='pause')
    # plt.title('spectogram %s, deltaf: %sHz' %tuple(ID_delta_f))
    # plt.xlim(times[0],times[-1])

fig = plt.figure(figsize = (8.27, 11.69/2))
ax0 = fig.add_subplot(221)
ax0.imshow(specs[0], cmap='jet', origin='lower', extent=(times[0], times[-1], sub_lim0[0], sub_lim1[0]), aspect='auto', vmin=-80, vmax=-10)
#ax0.plot(sub_times[0], jars[0], 'k', label = 'peak detection trace', lw = 2)
ax0.set_xlim(times[0],times[-1])
ax0.set_ylabel('frequency [Hz]')
ax0.axes.xaxis.set_ticklabels([])
ax0.set_title('∆F -2 Hz')
plt.xticks((1.7, 10, 20, 30, 40, 50, 60, times[-1]))

ax1 = fig.add_subplot(222)
ax1.imshow(specs[2], cmap='jet', origin='lower', extent=(times[0], times[-1], sub_lim0[2], sub_lim1[2]), aspect='auto', vmin=-80, vmax=-10)
#ax1.plot(sub_times[2], jars[2], 'k', label = 'peak detection trace', lw = 2)
ax1.set_xlim(times[0],times[-1])
ax1.axes.xaxis.set_ticklabels([])
#ax1.axes.yaxis.set_ticklabels([])
ax1.set_title('∆F 2 Hz')
ax1.get_shared_y_axes().join(ax0, ax1)
plt.xticks((1.7, 10, 20, 30, 40, 50, 60, times[-1]))

ax2 = fig.add_subplot(223)
ax2.imshow(specs[1], cmap='jet', origin='lower', extent=(times[0], times[-1], sub_lim0[1], sub_lim1[1]), aspect='auto', vmin=-80, vmax=-10)
#ax2.plot(sub_times[1], jars[1], 'k', label = 'peak detection trace', lw = 2)
ax2.set_xlim(times[0],times[-1])
ax2.set_ylabel('frequency [Hz]')
ax2.set_xlabel('time [s]')
ax2.set_title('∆F -10 Hz')
plt.xticks((1.7, 10, 20, 30, 40, 50, 60, times[-1]), [0, 10, 20, 30 ,40, 50, 60, 70])

ax3 = fig.add_subplot(224)
ax3.imshow(specs[3], cmap='jet', origin='lower', extent=(times[0], times[-1], sub_lim0[3], sub_lim1[3]), aspect='auto', vmin=-80, vmax=-10)
#ax3.plot(sub_times[3], jars[3], 'k', label = 'peak detection trace', lw = 2)
ax3.set_xlim(times[0],times[-1])
ax3.set_xlabel('time [s]')
#ax3.axes.yaxis.set_ticklabels([])
ax3.set_title('∆F 10 Hz')
plt.xticks((1.7, 10, 20, 30, 40, 50, 60, times[-1]), [0, 10, 20, 30 ,40, 50, 60, 70])
plt.subplots(sharex = True, sharey = True)
plt.show()

embed()

