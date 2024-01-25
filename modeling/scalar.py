import numpy as np
import matplotlib.pyplot as plt

class Wavefield_1D():
    
    def __init__(self):
        
        self._type = "1D wave propagation in constant density acoustic isotropic media"

        # TODO: read parameters from a file

        self.nt = 1001
        self.dt = 0.0012
        self.fmax = 20.0
       
        self.nz=1000
        self.dz=0.5
        self.tempo=np.zeros(self.nt)*self.dt
        self.depth=np.arange(self.nz)*self.dz
        self.c=343 #m/s
        self.interfaces = [300, 400, 600, 1000]
        self.velocities = [1500, 2000, 2500, 3000]
    def set_model(self):
        
    def get_type(self):
        print(self._type)

    def set_wavelet(self):
    
        t0 = 2.0*np.pi/self.fmax
        fc = self.fmax/(3.0*np.sqrt(np.pi))
        td = np.arange(self.nt)*self.dt - t0

        arg = np.pi*(np.pi*fc*td)**2.0

        self.wavelet = (1.0 - 2.0*arg)*np.exp(-arg)

    def plot_wavelet(self):
       
        t = np.arange(self.nt)*self.dt

        fig, ax = plt.subplots(figsize = (10, 5), clear = True)

        ax.plot(t, self.wavelet)
        ax.set_title("Wavelet", fontsize = 18)
        ax.set_xlabel("Time [s]", fontsize = 15)
        ax.set_ylabel("Amplitude", fontsize = 15) 
        
        ax.set_xlim([0, np.max(t)])
        
        fig.tight_layout()
        plt.show()


class Wavefield_2D(Wavefield_1D):
    
    def __init__(self):
        super().__init__()
        
        self._type = "2D wave propagation in constant density acoustic isotropic media"    


class Wavefield_3D(Wavefield_2D):
    
    def __init__(self):
        super().__init__()
        
        self._type = "3D wave propagation in constant density acoustic isotropic media"    

   