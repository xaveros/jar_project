import matplotlib.pyplot as plt
import numpy as np
import pylab
from IPython import embed
from scipy.optimize import curve_fit
from scipy.optimize import curve_fit
from matplotlib.mlab import specgram
import os

from jar_functions import import_data
from jar_functions import import_amfreq
from jar_functions import sin_response
from jar_functions import mean_noise_cut
from jar_functions import gain_curve_fit

#plt.rcParams.update({'font.size': 10})

def take_second(elem):      # function for taking the names out of files
    return elem[1]
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
              #'2020lepto16',
              #'2020lepto19',
              #'2020lepto20'
              ]
identifier_5Hz = [#'2018lepto1',
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
              #'2020lepto16',
              #'2020lepto19',
              #'2020lepto20'
              ]

for ident in identifier:

    times = []
    jars = []
    jms = []
    amfreq = []

    times1 = []
    jars1 = []
    jms1 = []
    amfreq1 = []

    amf = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

    data = sorted(np.load('5Hz_%s files.npy' %ident), key = take_second)      # list with filenames in it

    for i, d in enumerate(data):
        dd = list(d)
        if dd[1] == '1' or dd[1] == '0.2' or dd[1] == '0.05' or dd[1] == '0.01' or dd[1] == '0.005' or dd[1] == '0.001':
            jar = np.load('5Hz_%s.npy' %dd)     # load data for every file name
            jm = jar - np.mean(jar)         # low-pass filtering by subtracting mean

            time = np.load('5Hz_%s time.npy' %dd)       # time file
            dt = time[1] - time[0]

            n = int(1/float(d[1])/dt)
            cutf = mean_noise_cut(jm, n = n)
            cutt = time
            if dd[1] == '0.001':
                amfreq1.append(dd[1])
                jars1.append(jm - cutf)
                jms1.append(jm)
                times1.append(time)
            if dd[1] not in amfreq:
                print(dd)
                amfreq.append(dd[1])
                jars.append(jm - cutf)
                jms.append(jm)
                times.append(time)
            else:
                #print('1:', dd)
                amfreq1.append(dd[1])
                jars1.append(jm - cutf)
                jms1.append(jm)
                times1.append(time)
    if len(jars) != 6:
        continue

    sample = 100000

    fig = plt.figure(figsize=(8.27,11.69))
    fig.suptitle('%s' %ident)
    fig.text(0.06, 0.5, 'fish frequency [Hz]', ha='center', va='center', rotation='vertical', color = 'C0')
    fig.text(0.97, 0.5, 'stimulus amplitude [mV/cm]', ha='center', va='center', rotation='vertical', color = 'red')
    fig.text(0.5, 0.04, 'time [s]', ha='center', va='center')

    ax0 = fig.add_subplot(611)
    print('absolute frequency shift 0.001Hz:', np.max(jars[0]) - np.min(jars[0]))
    ax0.plot(times[0], jars[0], zorder = 20)
    #ax0.set_zorder(1)
    ax0.set_ylim(-12, 12)

    lower0 = 0
    upper0 = 2000
    x0 = np.linspace(lower0, upper0, sample)
    y0 = (sin_response(np.linspace(lower0, upper0, sample), 0.001, -np.pi/2, .35) + 0.5)
    ax0_0 = ax0.twinx()
    ax0_0.set_ylim(-0.2, 1.2)
    ax0_0.plot(x0, y0, color = 'red', zorder = 1, alpha = 0.5)
    #ax0_0.set_zorder(2)

    ax1 = fig.add_subplot(612)
    print('absolute frequency shift 0.005 Hz:', np.max(jars[1]) - np.min(jars[1]))
    ax1.plot(times[1], jars[1])
    ax1.set_ylim(-12, 12)

    lower1 = 0
    upper1 = 400
    x1 = np.linspace(lower1, upper1, sample)
    y1 = (sin_response(np.linspace(lower1, upper1, sample), 0.005, -np.pi / 2, .35) + 0.5)
    ax1_0 = ax1.twinx()
    ax1_0.set_ylim(-0.2, 1.2)
    ax1_0.plot(x1, y1, color='red', alpha = 0.5)

    ax2 = fig.add_subplot(613)
    print('absolute frequency shift 0.01 Hz:', np.max(jars[2]) - np.min(jars[2]))
    ax2.plot(times[2], jars[2])
    ax2.set_ylim(-12, 12)

    lower2 = 0
    upper2 = 400
    x2 = np.linspace(lower2, upper2, sample)
    y2 = (sin_response(np.linspace(lower2, upper2, sample), 0.01, np.pi / 2, -0.35) + 0.5)
    ax2_0 = ax2.twinx()
    ax2_0.set_ylim(-0.2, 1.2)
    ax2_0.plot(x2, y2, color='red', alpha = 0.5)

    ax3 = fig.add_subplot(614)
    print('absolute frequency shift 0.02 Hz:', np.max(jars[3]) - np.min(jars[3]))
    ax3.plot(times[3], jars[3])
    ax3.set_ylim(-12, 12)

    lower3 = 0
    upper3 = 200
    x3 = np.linspace(lower3, upper3, sample)
    y3 = (sin_response(np.linspace(lower3, upper3, sample), 0.05, np.pi / 2, -0.35) + 0.5)
    ax3_0 = ax3.twinx()
    ax3_0.set_ylim(-0.2, 1.2)
    ax3_0.plot(x3, y3, color='red', alpha = 0.5)

    ax4 = fig.add_subplot(615)
    print('absolute frequency shift 0.5 Hz:', np.max(jars[4]) - np.min(jars[4]))
    ax4.plot(times[4], jars[4])
    ax4.set_ylim(-12, 12)

    lower4 = 0
    upper4 = 200
    x4 = np.linspace(lower4, upper4, sample)
    y4 = (sin_response(np.linspace(lower4, upper4, sample), 0.2, np.pi / 2, -0.35) + 0.5)
    ax4_0 = ax4.twinx()
    ax4_0.set_ylim(-0.2, 1.2)
    ax4_0.plot(x4, y4, color='red', alpha = 0.5)

    ax5 = fig.add_subplot(616)
    print('absolute frequency shift 1 Hz:', np.max(jars[5]) - np.min(jars[5]))
    ax5.plot(times[5], jars[5])
    ax5.set_ylim(-12, 12)

    lower5 = 0
    upper5 = 200
    x5 = np.linspace(lower5, upper5, sample)
    y5 = (sin_response(np.linspace(lower5, upper5, sample), 1, np.pi / 2, -0.35) + 0.5)
    ax5_0 = ax5.twinx()
    ax5_0.plot(x5, y5, color='red', lw = 0.5, alpha = 0.5)
    ax5_0.set_ylim(-0.2, 1.2)
    '''
    
for ident in identifier_5Hz:

    times = []
    jars = []
    jms = []
    amfreq = []

    times1 = []
    jars1 = []
    jms1 = []
    amfreq1 = []

    amf = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    print('5Hz')
    data = sorted(np.load('5Hz_%s files.npy' % ident), key=take_second)  # list with filenames in it

    for i, d in enumerate(data):
        dd = list(d)
        if dd[1] == '1' or dd[1] == '0.2' or dd[1] == '0.05' or dd[1] == '0.01' or dd[1] == '0.005' or dd[1] == '0.001':
            jar = np.load('5Hz_%s.npy' % dd)  # load data for every file name
            jm = jar - np.mean(jar)  # low-pass filtering by subtracting mean

            time = np.load('5Hz_%s time.npy' % dd)  # time file
            dt = time[1] - time[0]

            n = int(1 / float(d[1]) / dt)
            cutf = mean_noise_cut(jm, n=n)
            cutt = time
            if dd[1] == '0.001':
                amfreq1.append(dd[1])
                jars1.append(jm - cutf)
                jms1.append(jm)
                times1.append(time)
            if dd[1] not in amfreq:
                #print(dd)
                amfreq.append(dd[1])
                jars.append(jm - cutf)
                jms.append(jm)
                times.append(time)
            else:
                #print('1:', dd)
                amfreq1.append(dd[1])
                jars1.append(jm - cutf)
                jms1.append(jm)
                times1.append(time)
    if len(jars) != 6:
        continue

    ax6 = fig.add_subplot(622)
    ax6.plot(times1[0], jars1[0])
    print('5Hz_absolute frequency shift 0.001Hz:', np.max(jms1[0]) - np.min(jms1[0]))
    ax6.set_ylim(-12, 12)
    ax6.axes.set_yticklabels([])
    plt.text(-0.05, 1.15, "B)", fontweight=550, transform=ax6.transAxes)

    ax7 = fig.add_subplot(624)
    ax7.plot(times1[1], jars1[1])
    print('5Hz_absolute frequency shift 0.005Hz:', np.max(jms1[1]) - np.min(jms1[1]))
    ax7.set_ylim(-12, 12)
    ax7.axes.set_yticklabels([])

    ax8 = fig.add_subplot(626)
    ax8.plot(times1[2], jars1[2])
    print('5Hz_absolute frequency shift 0.05Hz:', np.max(jms1[2]) - np.min(jms1[2]))
    ax8.set_ylim(-12, 12)
    ax8.axes.set_yticklabels([])

    ax9 = fig.add_subplot(6,2,8)
    ax9.plot(times1[3], jars1[3])
    print('5Hz_absolute frequency shift 0.02Hz:', np.max(jms1[3]) - np.min(jms1[3]))
    ax9.set_ylim(-12, 12)
    ax9.axes.set_yticklabels([])

    ax10 = fig.add_subplot(6,2,10)
    ax10.plot(times1[4], jars1[4])
    print('5Hz_absolute frequency shift 0.5Hz:', np.max(jms1[4]) - np.min(jms1[4]))
    ax10.set_ylim(-12, 12)
    ax10.axes.set_yticklabels([])

    ax11 = fig.add_subplot(6,2,12)
    ax11.plot(times1[5], jars1[5])
    print('5Hz_absolute frequency shift 1Hz:', np.max(jms1[5]) - np.min(jms1[5]))
    ax11.set_ylim(-12, 12)
    ax11.axes.set_yticklabels([])
    '''
    plt.subplots_adjust(left=0.125,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.35)
    plt.show()