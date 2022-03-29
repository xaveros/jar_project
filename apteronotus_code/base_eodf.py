import os
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
import nix_helpers as nh
from jar_functions import get_time_zeros
from jar_functions import parse_dataset
from jar_functions import mean_traces
from jar_functions import mean_noise_cut_eigen
from jar_functions import adjust_eodf

base_path = 'D:\\jar_project\\JAR\\eigenmannia\\sin'

identifier = ['2015eigen8',
              '2015eigen16','2015eigen17', '2015eigen19', '2015eigen15'
              # '2018lepto1',
              # '2018lepto4',
              # '2018lepto5',
              # '2018lepto76',
              # '2018lepto98',
              # '2019lepto03',
              # '2019lepto24',
              # '2019lepto27',
              # '2019lepto30',
              # '2020lepto04',
              # '2020lepto06',
              # '2020lepto16',
              # '2020lepto19',
              # '2020lepto20'
              ]
eod = []
for ID in identifier:
    base = []

    for dataset in os.listdir(os.path.join(base_path, ID)):
        if dataset == 'prerecordings':
            continue
        datapath = os.path.join(base_path, ID, dataset, 'beats-eod.dat')
        print(datapath)
        try:
            o = open(datapath)
        except IOError:
            continue
        frequency, time, amplitude, eodf, deltaf, stimulusf, duration, pause = parse_dataset(datapath)

        dm = np.mean(duration)
        pm = np.mean(pause)
        timespan = dm + pm
        start = np.mean([t[0] for t in time])
        stop = np.mean([t[-1] for t in time])

        mf, tnew = mean_traces(start, stop, timespan, frequency, time)  # maybe fixed timespan/sampling rate

        cf, ct = mean_noise_cut_eigen(mf, tnew, 1250)

        f = []
        for idx, i in enumerate(ct):
            if i > -45 and i < -5:
                f.append(cf[idx])
        ff = np.mean(f)
        base.append(ff)

        #plt.plot(ct, cf)
        #plt.show()
    base_eod = np.mean(base)
    print(ID)
    print(base_eod)
    eod.append(base_eod)

temp = np.load('temperature.npy')

eod_temp = zip(eod, temp)

Q10_eod = []
for et in eod_temp:
    Q10 = adjust_eodf(et[0], et[1])
    Q10_eod.append(Q10)

print('MAXI KING', np.max(Q10_eod))
print('MINI KING', np.min(Q10_eod))

embed()
