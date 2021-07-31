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
import seaborn as sns
from pylab import cycler

from scipy.spatial.distance import cosine, pdist, squareform
from scipy.stats import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y'])
pd.set_option('display.max_columns', None)

PATH = os.getcwd()
DIRECTORY = str(Path(PATH).parents[2])
PROJECT = str(Path(PATH).parents[1]).split('/')[-1]