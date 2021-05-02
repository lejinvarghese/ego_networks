import os
import sys
import numpy as np
import pandas as pd
import copy
import csv
import warnings
import operator
import tqdm
from pathlib import Path
from tqdm.notebook import trange, tqdm
from IPython.display import display, HTML

import matplotlib.pyplot as plt

from pylab import cycler

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Adobe Clean'
plt.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y'])
pd.set_option('display.max_columns', None)

_path = os.getcwd()
DIRECTORY = str(Path(_path).parents[2])
PROJECT = str(Path(_path).parents[1]).split('/')[-1]
