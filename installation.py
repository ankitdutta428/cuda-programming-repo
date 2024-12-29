import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
import os
import random 

def system_design():
    x = random.randint(1,1000)
    if(x%2):
        return True
    else:
        return False
    
