import cv2
import os
import numpy as np
from tqdm import tqdm
import h5py
import pyvista as pv
import napari
from scipy.ndimage import binary_dilation
import pmcx
import numpy as np
import matplotlib.pyplot as plt


class depth2cube():
    
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
            
            # Creating the cube
            cube = []
            for n in range(255, -1, -1):
                temp = np.zeros(np.shape(img), 'uint8')
                temp[img==n] = 1
                cube.append(temp)
            cube = np.stack(cube, axis=2)

            ## Dilating the cube
            # Define the structure for dilation
            structure = np.ones((3, 3, 3))
            # Apply binary dilation
            cube = binary_dilation(cube, structure=structure).astype('uint8')
            cube[cube==1] = 255

            if save:
                with h5py.File(os.path.join(self.dir_output, name), 'w') as f:
                    f.create_dataset('voxels', data=cube, compression='gzip')
            
            if show=='3D':
                # Create a function to scale the image to 0-255 range (future)
                viewer = napari.Viewer()
                viewer.add_image(cube, name='3D Image Cube')
                napari.run()
            
            # If 'show' is True, plot the 3D image based on the 2D image
            if show=='2D':
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

# Segmentar uma função somente para imprimir 2D depth 2 contorno (show='2D')



def mcx_run(**kwargs):
    
    # First, creating the configuration file
    show_image = kwargs.get('show_image', True)
    save_folder = kwargs.get('save_folder', False)
    
    cfg={}
    
    cfg['nphoton'] = kwargs.get('nphoton', 1e6)
    cfg['vol'] = kwargs.get('vol', np.ones([60,60,60], dtype='uint8'))
    cfg['tstart'] = kwargs.get('tstart', 0)
    cfg['tend'] = kwargs.get('tend', 5e-9)
    cfg['tstep'] = kwargs.get('tstep', 5e-9)
    cfg['srcpos'] = kwargs.get('srcpos', [30, 30, 0])
    cfg['srcdir'] = kwargs.get('srcdir', [0, 0, 1])
    
    # The properties are in the shape [mua, mus, g, n]
    prop_back = kwargs.get('prop_back', [0, 0, 1, 1])
    cfg['prop'] = kwargs.get('prop', False)
    
    
    if not cfg['prop']:
        cfg['prop'] = prop_back
        
    
    
    cfg['prop'] = [
        [0, 0, 1, 1],
        [0.005, 1, 0.01, 1.37]]
    
    # First, verifying if there is GPU
    try:
        pmcx.gpuinfo()
    except:
        pass
    
    # Actually running the simulation
    res=pmcx.run(cfg)
    print('\n\nres= ', print(type(res)), print(res))
    res.keys()
    print('\n\nres= ', print(type(res)), print(res))
    # Acquiring the flux
    res['flux'].shape
    
    if show_image:
        plt.subplots()
        plt.imshow(np.log10(res['flux'][30,:,:]))
        plt.title('Logarithm of the Radiant Flux in base 10')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    
    if save_folder:
        os.chdir(save_folder)
        image = np.log10(res['flux'][30,:,:])
        cv2.imwrite('image.png', image)
        




if __name__ == "__main__":
    # Choose a directory with the images to predicth depth
    dir_input = r'C:\Users\marlo\Downloads\MiDas\2024.05.24 - Testes Iniciais\depth images'
    # Choose a directory to save predicted images
    dir_output = r'C:\Users\marlo\Downloads\MiDas\2024.05.24 - Testes Iniciais\hypercubes'
    
    d2c = depth2cube(dir_input, dir_output)
    
    d2c.main(div=3, show='3D', save=False)
