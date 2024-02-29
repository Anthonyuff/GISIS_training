import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.parametro=[-2,-1]
       
       
        self.x=np.linspace(-2,2,101)
        
        self.y = self.parametro[0] + self.parametro[1] * self.x
        self.y_n = self.y + np.random.rand(len(self.y))
       
       
        #CMP
        self.v_true=4000
        self.z_true= 800
        
        self.p0=((2*self.z_true)/self.v_true)**2
        self.p1=(1/self.v_true)**2
      
        self.nx=5000
        self.offset= np.linspace(25,self.nx,700)
        self.gp=np.sqrt(self.p0 + self.p1*(self.offset)**2)
        self.gp_n=self.gp + 5e-2*(0.05-np.random.rand(len(self.gp)))
       
        self.modelo=solution((self.offset)**2,(self.gp_n)**2)
        self.gp2=np.sqrt(self.modelo[0] + self.modelo[1]*(self.offset)**2)
        

        
       

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
                self.y_p = self.a[i, j] + self.b[i, j] * self.offset #self.x para equacao linear
                self.mat[i, j] = np.sqrt(np.sum((self.gp_n - self.y_p)**2)) #self.y_n para equacao linear
        #self.min_index = np.unravel_index(np.argmin(self.mat, axis=None), self.mat.shape)
        self.a0_ind, self.a1_ind = np.where(self.mat == np.min(self.mat))
    
        
        
    def plot_imshow(self):    
        
        fig ,ax = plt.subplots()



        ax.imshow(self.mat,extent=[-5,5,5,-5])
        
        ax.scatter( self.a[self.a0_ind, self.a1_ind],self.b[self.a0_ind, self.a1_ind])
        print(np.min(self.mat))
        fig.tight_layout()
        plt.show()



class CMP(LinearRegression):
    
        



    def plot_cmpruido(self):

        fig , ax=plt.subplots(figsize=(8, 5), clear=True) 
      
        ax.plot(self.offset,self.gp_n,'r')
        ax.plot(self.offset,self.gp2)
        fig.tight_layout()
        ax.invert_yaxis()
        plt.show()

    


def solution(x,noise):
    G = np.zeros((len(x), 2))

    for n in range(2): 
                G[:,n] = x**n

    GTG = np.dot(G.T, G)
    GTD = np.dot(G.T,noise)

    m = np.linalg.solve(GTG, GTD)
    return m
    
