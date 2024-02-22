from function import LinearRegression

import numpy as np
import matplotlib.pyplot as plt
def simulation ():
    model = LinearRegression()
    model.reta()
    model.ruido()

    model.solution_space()

if __name__ == "__main__":
    simulation()