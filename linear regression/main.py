from function import LinearRegression,LinearRegressionIMSHOW

import numpy as np
import matplotlib.pyplot as plt
def simulation ():
    model = LinearRegression()
    model.plot_reta()
    model.plot_ruido()
    man= LinearRegressionIMSHOW()
    man.solution_space()
    man.plot_imshow()

if __name__ == "__main__":
    simulation()