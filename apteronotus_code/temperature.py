import os
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from jar_functions import parse_infodataset

base_path = 'D:\\jar_project\\JAR\\sin'

identifier = ['2018lepto1',
              '2018lepto4',
              '2018lepto5',
              '2018lepto76',
              '2018lepto98',
              '2019lepto03',
              '2019lepto24',
              '2019lepto27',
              '2019lepto30',
              '2020lepto04',
              '2020lepto06',
              '2020lepto16',
              '2020lepto19',
              '2020lepto20']

av_temperature = []
for ID in identifier:
    temperature = []
    datapath = os.path.join(base_path, ID)
    for dataset in os.listdir(datapath):
        if dataset == 'prerecordings':
            continue
        data = os.path.join(datapath, dataset, 'info.dat')
        #print(data)
        i, temp = parse_infodataset(data)
        temperature.append(float(temp[0]))
    print(i)
    print(np.mean(temperature))
    av_temperature.append(np.mean(temperature))
np.save('temperature.npy', av_temperature)
embed()