import matplotlib.pyplot as plt
import numpy as np
import pylab
from IPython import embed
from scipy.optimize import curve_fit
from jar_functions import gain_curve_fit
from jar_functions import avgNestedLists
import matplotlib as mpl
from matplotlib import cm

ab = []
a = [1, 1, None, 1]
b = [2, 2, 2, 2]
ab.append(a)
ab.append(b)
print(ab)
print(np.mean(ab, axis = 0))
#av = avgNestedLists(np.array(ab))
#print(av)

