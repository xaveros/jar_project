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

plt.rcParams.update({'font.size': 12})

base_path = 'D:\\jar_project\\JAR\\eigenmannia\\deltaf'

#2015eigen8 no nix files
identifier = ['2013eigen13',
              '2015eigen16','2015eigen17', '2015eigen19', '2020eigen22','2020eigen32']


response = []
deltaf = []
for ID in identifier:
    for dataset in os.listdir(os.path.join(base_path, ID)):
        datapath = os.path.join(base_path, ID, dataset, '%s.nix' % dataset)
        print(datapath)
        stimuli_dat = os.path.join(base_path, ID, dataset, 'manualjar-eod.dat')
        #print(stimuli_dat)
        delta_f, duration = parse_stimuli_dat(stimuli_dat)
        dur = int(duration[0][0:2])
        print(delta_f)
        #if delta_f != [-4.0]:
        #   continue
        data, pre_data, dt = import_data_eigen(datapath)

        #hstack concatenate: 'glue' pre_data and data
        dat = np.hstack((pre_data, data))

        # data
        nfft = 2**17
        spec, freqs, times = specgram(dat[0], Fs=1 / dt, detrend='mean', NFFT=nfft, noverlap=nfft * 0.95)
        dbspec = 10.0 * np.log10(spec)  # in dB
        power = dbspec[:, 25]

        fish_p = power[(freqs > 200) & (freqs < 1000)]
        fish_f = freqs[(freqs > 200) & (freqs < 1000)]

        index = np.argmax(fish_p)
        eodf = fish_f[index]
        eodf4 = eodf * 4

        lim0 = eodf4 - 40
        lim1 = eodf4 + 60

        df = freqs[1] - freqs[0]
        ix0 = int(np.floor(lim0/df))    # back to index
        ix1 = int(np.ceil(lim1/df))    # back to index
        spec4= dbspec[ix0:ix1, :]
        freq4 = freqs[ix0:ix1]
        jar4 = freq4[np.argmax(spec4, axis=0)]      # all freqs at max specs over axis 0

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

        r = (np.median(j) - np.median(b)) / 4       # divided by 4 cause of data at 4th harmonic, therefore response 4 times higher
        print('response:', r)
        deltaf.append(delta_f[0])
        response.append(r)

        plt.figure(figsize = (8.27,11.69/2))
        plt.imshow(spec4, cmap='jet', origin='lower', extent=(times[0], times[-1], lim0, lim1), aspect='auto', vmin=-80, vmax=-10)
        plt.plot(cut_time_jar, jar4, color = 'k', label = 'peak detection trace', lw = 2)
        plt.hlines(y=lim0 + 5, xmin=0, xmax=10, lw=4, color='red', label='pause')
        plt.hlines(y=lim0 + 5, xmin=10, xmax=70, lw=4, color='gold', label='stimulus duration')
        plt.title('spectogram %s, deltaf: %sHz' %tuple(ID_delta_f))
        plt.xlim(times[0],times[-1])
        #embed()
        plt.xticks((times[0], 10, 20, 30, 40, 50, 60, times[-1]), [0, 10, 20, 30 ,40, 50, 60, 70])
        plt.xlabel('time [s]')
        plt.ylabel('frequency [Hz]')
        plt.legend(loc = 'best')
        #plt.show()
        delta_f_ID = [str(delta_f[0]).split('.')[0], ID]

        plt.close()


    res_df = sorted(zip(deltaf,response))

    np.save('res_df_%s_new' %ID, res_df)
