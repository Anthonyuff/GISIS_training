from function import LinearRegression,LinearRegressionIMSHOW,CMP


import numpy as np
import matplotlib.pyplot as plt
def simulation ():
    model = LinearRegression()
    model.plot_reta()
    model.plot_ruido()
    man= LinearRegressionIMSHOW()
    man.solution_space()
    man.plot_imshow()
    
    

    cmp=CMP()
    cmp.plot_cmpruido()

if __name__ == "__main__":
    simulation()