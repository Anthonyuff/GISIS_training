import numpy as np
import matplotlib.pyplot as plt

from numba import njit, prange 
from matplotlib.animation import FuncAnimation





class Wavefield_1D():
    
    def __init__(self):
        
        self._type = "1D wave propagation in constant density acoustic isotropic media"

        # TODO: read parameters from a file

        self.nt = 10001
        self.dt = 0.0012
        self.fmax = 20.0
       
        self.nz=1000
        self.dz=7
        self.tempo=np.arange(self.nt)*self.dt
        self.depth=np.arange(self.nz)*self.dz
        
        self.interfaces = [1000, 2000, 3000, 4000]
       
        #self.velocities = [1500, 2000, 2500, 3000,4500]
        self.velocities = [2000, 2000, 2000, 2000,2000] 
       
        self.model = np.full(self.nz, self.velocities[0])
        self.z_fonte=[100,300,500]
        self.z_recp=[800,1000,2000,3500,4000]
        
        self.fig, self.ax = plt.subplots(num="Wavefield plot", figsize=(8, 8), clear=True)
        
        

        
    def set_model(self):#configurar a velocidade com a interface
        
        
        self.interface_indices = np.searchsorted(self.depth, self.interfaces)
        
        for layerId, index in enumerate(self.interface_indices):
            self.model[index:] = self.velocities[layerId+1]
            print(self.model)


    def plot_model(self):
        
        plt.plot(self.model,self.depth)
        plt.title("Model", fontsize = 18)
        plt.xlabel("Velocity [m/s]", fontsize = 15)
        plt.ylabel("Depth [m]", fontsize = 15)
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.scatter(np.full(len(self.z_fonte),self.velocities[0]), self.z_fonte, color='red', marker='*', label='Fonte')
        plt.scatter(self.velocities, self.z_recp, color='blue', marker='v', label='Receptor')
    

        plt.tight_layout()
        plt.legend()


        plt.savefig('modelo_de_velocidade.png',  bbox_inches='tight')
        plt.show()

    def wave_prog(self):
        self.P = np.zeros((self.nz, self.nt)) # P_{i,n}

        sId = int(self.z_fonte[0] / self.dz)

        for n in range(1,self.nt-1):

            self.P[sId,n] += self.wavelet[n]    

            laplacian = laplacian_1d(self.P, self.dz, self.nz, n)

            self.P[:,n+1] = (self.dt*self.model)**2 * laplacian + 2.0*self.P[:,n] - self.P[:,n-1]
        
       

    def get_type(self):
        print(self._type)

    

    def plot_wavefield(self):
        
        fig2 , ax = plt.subplots(num="Wavefield plot", figsize=(8, 8), clear=True)

        #ax.plot(self.P[:,5000])
        
        ax.imshow(self.P, aspect = "auto", cmap = "Greys")

        

        ax.set_title("Wavefield", fontsize = 18)
        ax.set_xlabel("Time [s]", fontsize = 15)
        ax.set_ylabel("Depth [m]", fontsize = 15) 
        
        fig2.tight_layout()
        plt.show()
    
    def grafico_animação(self):
        fig,ax = plt.subplots(num="Wavefield plot", figsize=(8, 8), clear=True)
        
        wave, = ax.plot(self.P[:,0],self.depth)
        ax.set_title("Campo de Ondas", fontsize=18)
        ax.set_xlabel("Tempo [s]", fontsize=15)
        ax.set_ylabel("Profundidade [m]", fontsize=15)
        fig.tight_layout()
        ax.set_ylim([0, (self.nz-1)*self.dz])
        ax.set_xlim([np.min(self.P)-0.5, np.max(self.P)+0.5])
        ax.invert_yaxis()
        grafico = []

        grafico.append(wave)
        
    
        

     

        def init():
            wave.set_ydata(self.depth)  
            wave.set_xdata([np.nan] * self.nz)
                        
            return grafico

        def animate(frames): 
            wave.set_ydata(self.depth)  
            wave.set_xdata(self.P[:,frames])
        
             

            return grafico
        
    
        
        _= FuncAnimation(fig, animate,init_func=init ,frames=self.nt, interval=1e-3*self.dt, blit=True)
        _.save('animacao.gif', writer='pillow')
        plt.show()  
        

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
    
@njit 
def laplacian_1d(P , dz, nz, time_id):
    d2P_dz2 = np.zeros(nz)

    for i in prange(1, nz-1): 
        d2P_dz2[i] = (P[i-1,time_id] - 2.0*P[i,time_id] + P[i+1,time_id]) / dz**2.0    

    return d2P_dz2

class Wavefield_2D(Wavefield_1D):
    
    def __init__(self):
        super().__init__()
        
        self._type = "2D wave propagation in constant density acoustic isotropic media"    


class Wavefield_3D(Wavefield_2D):
    
    def __init__(self):
        super().__init__()
        
        self._type = "3D wave propagation in constant density acoustic isotropic media"    

   