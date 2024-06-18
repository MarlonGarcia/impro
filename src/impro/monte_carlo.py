import cv2
import os
import numpy as np
from tqdm import tqdm
import h5py
import pyvista as pv

class depth2contour():
    
    def __init__(self, dir_input, dir_output):
        self.dir_input = dir_input
        self.dir_output = dir_output
    
    def main(self, **kwargs):
        div = kwargs.get('div', 1)
        output = kwargs.get('output', False)
        save = kwargs.get('save', True)
        show = kwargs.get('show', False)
        
        if output:
            cubes = []
        
        name = 'Initial data..'
        loop = tqdm(os.listdir(self.dir_input), desc=f'\nProcessing {name}')
        for name in loop:
            img = cv2.imread(os.path.join(self.dir_input, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if div != 1:
                img = cv2.resize(img, (int(np.shape(img)[1]/div),
                                       int(np.shape(img)[0]/div)))
            
            if (output+save):
                cube = []
                for n in range(256, -1, -1):
                    temp = np.zeros(np.shape(img), 'uint8')
                    temp[img==n] = 1
                    cube.append(temp)
                cube = np.stack(cube, axis=2)
                cube[cube==1] = 255
                if save:
                    with h5py.File(os.path.join(self.dir_output, name), 'w') as f:
                        f.create_dataset('voxels', data=cube, compression='gzip')
            
            if show:
                xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
                z = img.astype(float)
                # Create the PyVista structured grid
                mesh = pv.StructuredGrid(xx, yy, z)
                # Create the plotter object
                plotter = pv.Plotter()
                # Add the mesh to the plotter
                plotter.add_mesh(mesh, cmap='gray', opacity=0.7)
                # Adjust the camera to get a better view
                plotter.view_yz()
                plotter.camera_position = 'xy'
                plotter.camera.elevation = 90  # Adjust the camera elevation to make the plot more visible
                plotter.camera.zoom(1.5)  # Adjust zoom level for better view
                # Setting the plotter title
                plotter.add_title("3D Depth Image Visualization")
                # Show the plotter window and pause the script until it's closed
                plotter.show(interactive=True)
            if output:
                cubes.append(cube)
            
        if output:
            return cubes


# Adquirir imagens, segmentar no que deve ir ao monte carlo e o que é fundo
# Segmentar os tecidos requeridos, e ir ao monte carlo com o mua e mus
# Documentar melhor o 'esc' na função


if __name__ == "__main__":
    # Choose a directory with the images to predicth depth
    dir_input = r'C:\Users\marlo\Downloads\MiDas\2024.05.24 - Testes Iniciais\depth images'
    # Choose a directory to save predicted images
    dir_output = r'C:\Users\marlo\Downloads\MiDas\2024.05.24 - Testes Iniciais\hypercubes'
    
    d2c = depth2contour(dir_input, dir_output)
    
    d2c.main(div=3, show=True, save=False)
