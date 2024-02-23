import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.parametro=[-2,-1]
       
       
        self.x=np.linspace(-2,2,101)
        
        self.y = self.parametro[0] + self.parametro[1] * self.x
        self.y_n = self.y + np.random.rand(len(self.y))
       

    # Função que descreve a reta
    def plot_reta(self):
        
        fig,ax= plt.subplots()
        ax.plot(self.x,self.y)
        fig.tight_layout()
        plt.show()
	
	
	    

    # Aplicar o ruído no eixo y
    def plot_ruido(self):
        
        fig,ax= plt.subplots()
        ax.plot(self.x,self.y_n)
        fig.tight_layout()
        plt.show()

    # Visualização da reta
    

    # Criar espaço solução com vários coeficientes a0 e a1
    # Correlacionar através da norma L2 a diferença
class LinearRegressionIMSHOW(LinearRegression):   
    def solution_space(self):
        n = 1001

        self.a = np.linspace(-5, 5, n)
        self.b = np.linspace(-5, 5, n)

        self.a, self.b = np.meshgrid(self.a, self.b)

        self.mat = np.zeros([n, n])
        
        
        

        for i in range(n):
            for j in range(n):
                self.y_p = self.a[i, j] + self.b[i, j] * self.x
                self.mat[i, j] = np.sqrt(np.sum((self.y_n - self.y_p)**2))
        #self.min_index = np.unravel_index(np.argmin(self.mat, axis=None), self.mat.shape)
        self.a0_ind, self.a1_ind = np.where(self.mat == np.min(self.mat))
    
        
        
    def plot_imshow(self):    
        
        fig ,ax = plt.subplots()



        ax.imshow(self.mat,extent=[-5,5,5,-5])
        
        ax.scatter( self.a[self.a0_ind, self.a1_ind],self.b[self.a0_ind, self.a1_ind])
        print(np.min(self.mat))
        fig.tight_layout()
        plt.show()
class MMQ(LinearRegression):
    def solution(self):
        G = np.zeros((len(self.y_n), len(self.parametro)))

        for n in range(len(self.parametro)): 
                G[:,n] = self.x**n

        GTG = np.dot(G.T, G)
        GTD = np.dot(G.T, self.y_n)

        self.m = np.linalg.solve(GTG, GTD)
        print(self.m)
