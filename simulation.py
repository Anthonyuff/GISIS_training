from modeling import scalar

def simulation():
    #abstração(base da computação )
    #herança
    #encapsulamento(proteção/segurança)
    #polimorfismo 
    id = 0

    myWave = [scalar.Wavefield_1D(), 
              scalar.Wavefield_2D(),
              scalar.Wavefield_3D()] 

    # print(myWave[id]._type)
   
  

    myWave[id].set_model()
    myWave[id].set_wavelet()
    myWave[id].wave_prog()
    myWave[id].plot_wavefield()
    myWave[id].plot_model()
    


if __name__ == "__main__":
    simulation()

