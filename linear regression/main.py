from function import LinearRegression,LinearRegressionIMSHOW,MMQ

import numpy as np
import matplotlib.pyplot as plt
def simulation ():
    model = LinearRegression()
    model.plot_reta()
    model.plot_ruido()
    man= LinearRegressionIMSHOW()
    man.solution_space()
    man.plot_imshow()
    mmq=MMQ()
    mmq.solution()

if __name__ == "__main__":
    simulation()