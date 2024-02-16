import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.a0= -2
        self.a1= -1
        self.x=np.linspace(-2,2,101)
       

    # Função que descreve a reta
    def reta(self, x):
        self.y = self.a0 + self.a1 * x
        fig,ax= plt.subplots()
        ax.plot(self.x,self.y)
        fig.tight_layout()
        plt.show()
	
	
	    

    # Aplicar o ruído no eixo y
    def ruido(self):
        self.y_n = self.y + np.random.rand(len(self.y))
        fig,ax= plt.subplots()
        ax.plot(self.x,self.y_n)
        fig.tight_layout()
        plt.show()

    # Visualização da reta
    

    # Criar espaço solução com vários coeficientes a0 e a1
    # Correlacionar através da norma L2 a diferença
    def solution_space(self, x, y):
        n = 1001
        self.a0 = np.linspace(-4, 4, n)
        self.a1 = np.linspace(-5, 5, n)

        self.a0, self.a1 = np.meshgrid(self.a0, self.a1)

        self.mat = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                self.y_p = self.a0[i, j] + self.a1[i, j] * x
                self.mat[i, j] = np.sqrt(np.sum((y - self.y_p)**2))
        fig,ax = plt.subplot()
        ax.imshow(self.mat,extent=[-5,5,-5,5])
        plt.show()

# Exemplo de uso

