''' 
Created by Hayoung Chung, 
@Date: Apr 24 2023
'''

'''
This is the main file for the Q-learning algorithm using the Bellman equation.
The original code is from the paper "Deep Reinforcement Learning..." by Nathan Brown
'''

import time
import numpy as np
import matplotlib.pyplot as plt 
import sys
import random as rand
import warnings
import skimage.measure as measure
import pandas as pd
import pickle
if not sys.warnoptions:
    warnings.simplefilter("ignore")
tic=time.perf_counter()

import finite_element as fea

# Define the size and scope 
Elements_X=4
Elements_Y=4
ElementSize=Elements_X*Elements_Y
Vol_fraction=10./16.
Remove_num=ElementSize-(ElementSize*Vol_fraction)
