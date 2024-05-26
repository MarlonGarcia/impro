import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

class depth2contour():
    
    def __init__(self, dir_input, dir_output):
        self.dir_input = dir_input
        self.dir_output = dir_output
    
    def main(self):
        for name in os.listdir(self.dir_input):
            
            img = cv2.imread(os.path.join(self.dir_input, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            output = []
            loop = tqdm(range(255, -1, -1), desc=f'Image {name}')
            for n in loop:
                temp = np.zeros(np.shape(img), 'uint8')
                temp[img==n] = 1
                output.append(temp)
            output = np.stack(output, axis=2)
            print(np.shape(output))
            
            dim = np.shape(output)
            X, Y, Z = np.meshgrid(np.arange(dim[0]), np.arange(dim[1]), np.arange(dim[2]))

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, facecolors=output.reshape(-1), cmap='gray')
            ax.set_xlabel('X')
            
            break
        return output
    

# Adquirir imagens, segmentar no que deve ir ao monte carlo e o que é fundo
# Segmentar os tecidos requeridos, e ir ao monte carlo com o mua e mus
# Documentar melhor o 'esc' na função


if __name__ == "__main__":
    # Choose a directory with the images to predicth depth
    dir_input = r'C:\Users\marlo\Downloads\MiDas\2024.05.24 - Testes Iniciais\depth images'
    # Choose a directory to save predicted images
    dir_output = r'C:\Users\marlo\Downloads\MiDas\2024.05.24 - Testes Iniciais\hypercubes'
    
    d2c = depth2contour(dir_input, dir_output)
    
    output = d2c.main()
