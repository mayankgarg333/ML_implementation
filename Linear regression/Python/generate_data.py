# This file generate noisy data for linear regression problem 

import os
import numpy as np


for points in range(10,101,10):
    x=np.linspace(0,4*np.pi,points)
    y=np.sin(.5*x)
    
    mean=0
    std=.3
    noise=mean+np.random.normal(mean, std,x.shape[0])
    t=y+noise
    
    directory="data"
    
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    np.savez('data/data'+str(points), x=x,y=y,t=t)
    
# To load use locals().update(np.load(filename))
