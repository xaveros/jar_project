import matplotlib.pyplot as plt
import matplotlib as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
import glob
import IPython
import numpy as np
from IPython import embed
from scipy.optimize import curve_fit
from jar_functions import parse_dataset
from jar_functions import parse_infodataset
from jar_functions import mean_traces
from jar_functions import mean_noise_cut_eigen
from jar_functions import norm_function
from jar_functions import step_response
from jar_functions import sort_values
from jar_functions import average

base_path = 'D:\\jar_project\\JAR\\eigen\\step'

identifier = ['step_2015eigen8',
              'step_2015eigen15\\+15Hz',
              'step_2015eigen16',
              'step_2015eigen17',
              'step_2015eigen19']
datasets = []

#dat = glob.glob('D:\\jar_project\\JAR\\2020*\\beats-eod.dat')
#infodat = glob.glob('D:\\jar_project\\JAR\\2020*\\info.dat')

time_all = []
freq_all = []

ID = []
col = ['dimgrey', 'grey', 'darkgrey', 'silver', 'lightgrey', 'gainsboro', 'whitesmoke']
labels = zip(ID, datasets)

for infodataset in datasets:
    infodataset = os.path.join(base_path, infodataset, 'info.dat')
    i = parse_infodataset(infodataset)
    identifier = i[0]
    ID.append(identifier)

for ID in identifier:
    base_path = 'D:\\jar_project\\JAR\\eigenmannia\\step\\%s' %ID
    response = []
    stim_ampl = []
    for idx, dataset in enumerate(os.listdir(base_path)):
        data = os.path.join(base_path, dataset, 'beats-eod.dat')

        if dataset == 'prerecordings':
            continue
        #input of the function
        frequency, time, amplitude, eodf, deltaf, stimulusf, stimulusamplitude, duration, pause = parse_dataset(data)
        dm = np.mean(duration)
        pm = np.mean(pause)
        timespan = dm + pm
        start = np.mean([t[0] for t in time])
        stop = np.mean([t[-1] for t in time])

        print(dataset)

        mf, tnew = mean_traces(start, stop, timespan, frequency, time) # maybe fixed timespan/sampling rate

        cf, ct = mean_noise_cut_eigen(mf, tnew, n=1250)

        onset_point = dm - dm
        offset_point = dm
        onset_end = onset_point - 10
        offset_start = offset_point - 10


        b = []
        for index, i in enumerate(ct):
            if i > -45 and i < -5:
                b.append(cf[index])
        j = []
        for indexx, h in enumerate(ct):
            if h < 195 and h > 145:
                j.append(cf[indexx])

        r = np.median(j) - np.median(b)
        response.append(r)
        stim_ampl.append(float(stimulusamplitude[0]))

    res_ampl = sorted(zip(stim_ampl, response))

    plt.plot(stim_ampl, response, 'o')
    plt.xlabel('Stimulusamplitude')
    plt.ylabel('absolute JAR magnitude')
    plt.title('absolute JAR')
    plt.xticks(np.arange(0.0, 0.3, step=0.05))
    #plt.savefig('relative JAR')
    #plt.legend(loc = 'lower right')
    plt.show()
embed()


# natalie fragen ob sie bei verschiedenen Amplituden messen kann (siehe tim)
