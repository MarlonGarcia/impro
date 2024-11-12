# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
# from tqdm import tqdm
import pmcx
import matplotlib.pyplot as plt


def mcx_run(**kwargs):
    """
    Function to run a Monte Carlo simulation of light-tissue interactions using
    the library Monte Carlo eXtreme (GitHub: https://github.com/fangq/mcx)

    Parameters
    ----------
    **kwargs : dict, optional
        Optional keyword arguments for configuring the simulation:
        
        - `show_image` (bool): Whether to display a logarithmic image of the radiant flux in the middle of the volume.
          Defaults to `True`.
        - `save_folder` (str or bool): If set to a valid folder path, saves the generated radiant flux image to this location.
          Defaults to `False` (no saving).
        - `prop` (list of floats): Optical properties for the tissue, provided as `[mua, mus, g, n]`. Defaults to
          `[0.01, 0.2, 0.9, 1.37]`, which are example values for skin epidermis.
        - `prop_background` (list of floats): Optical properties for the background medium, provided as `[mua, mus, g, n]`.
          Defaults to `[0, 0, 1, 1]`.
        - `cfg` (dict): Configuration dictionary for the simulation. The following keys can be defined:
            - `nphoton` (float): Number of photons to simulate. Default is `1e6`.
            - `vol` (3D numpy array): Volume array representing the tissue, where each element defines the medium type
              at that voxel. Defaults to a cube of ones with dimensions `[60, 60, 60]`.
            - `tstart` (float): Start time for the simulation in seconds. Default is `0`.
            - `tend` (float): End time for the simulation in seconds. Default is `5e-9`.
            - `tstep` (float): Time step for the simulation in seconds. Default is `5e-9`.
            - `srcpos` (list of ints): Source position in the volume, given as `[x, y, z]` coordinates. Defaults to the center of the XY plane at `z = 0`.
            - `srcdir` (list of ints): Source direction vector. Defaults to `[0, 0, 1]` (along the Z axis).
            - `prop` (2D numpy array): Optical properties stacked as rows for each medium type. If not provided, this is set by stacking `prop_background` and `prop`.
          
          OBS: For more informtion about the configuration, please visit the Matlab documentation (the `cfg` variable follow the same parameters) at https://github.com/fangq/mcx/tree/master/mcxlab

    Returns
    -------
    res : dict
        Dictionary containing simulation results. The `flux` field in `res` stores the 4D numpy array of radiant flux values, with dimensions `[X, Y, Z, T]`, where `X`, `Y`, and `Z` are the spatial dimensions and `T` is time.

    Raises
    ------
    ValueError
        If a compatible GPU is not detected or if the GPU driver version is insufficient for the `pmcx` library.

    Notes
    -----
    - The `pmcx` library requires a GPU for simulations. Ensure your system has compatible GPU hardware and drivers.
    - If `show_image` is enabled, a plot is displayed showing the logarithm of the radiant flux in the middle of the
      volume along the Z-axis.
    - If `save_folder` is set to a valid path, the logarithmic image of the radiant flux is saved to `image.png` in
      the specified folder.

    Example
    -------
    ```python
    # Define the configuration dictionary
    cfg = {
        'nphoton': 1e7,
        'vol': np.ones([100, 100, 100], dtype='uint8')
    }
    # Run the simulation
    res = mcx_run(cfg=cfg)
    
    # Visualize the radiant flux in the middle of the cube along the Z-axis
    cube = res['flux'][..., 0]
    plt.plot(cube[30, 30, :])
    plt.show()
    ```
    """

    # First, acquiring `kwargs`, the user optional inputs
    show_image = kwargs.get('show_image', True)
    save_folder = kwargs.get('save_folder', False)
    # Defining optical properties, in this order: [mua, mus, g, n]
    # If user does not define, we use test values "for skin epidermis"
    prop = kwargs.get('prop', [0.01, 0.2, 0.9, 1.37])
    prop_background = kwargs.get('prop_background', [0, 0, 1, 1])
    
    cfg = kwargs.get('cfg', {})
    cfg['nphoton'] = cfg.get('nphoton', 1e6)
    cfg['vol'] = cfg.get('vol', np.ones([60, 60, 60], dtype='uint8'))
    cfg['tstart'] = cfg.get('tstart', 0)
    cfg['tend'] = cfg.get('tend', 5e-9)
    cfg['tstep'] = cfg.get('tstep', 5e-9)
    # If user does not define, the source is positioned in the middle of the XY plane
    cfg['srcpos'] = cfg.get('srcpos', [int(np.shape(cfg['vol'])[0]/2),
                                       int(np.shape(cfg['vol'])[1]/2), 0])
    # Source direction is defined to the Z axis, if user does not define it
    cfg['srcdir'] = cfg.get('srcdir', [0, 0, 1])
    
    # Defining optical properties by stacking background with tissue properties:
    cfg['prop'] = cfg.get('prop', np.vstack((prop_background, prop)))
    
    # Second, verifying if there is GPU available
    try:
        pmcx.gpuinfo()
    except:
        raise ValueError('The GPU drive version is not sufficient for this application, or no GPU was found.')
    
    # Actually running the simulation
    res = pmcx.run(cfg)
    
    # Printing the irradiance in the middle of the cube, if user choose it
    if show_image:
        plt.subplots()
        plt.imshow(np.log10(res['flux'][int(np.shape(cfg['vol'])[0]/2), :, :, 0]))
        plt.title('Logarithm of the Radiant Flux in base 10')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    
    if save_folder:
        os.chdir(save_folder)
        image = np.log10(res['flux'][int(np.shape(cfg['vol'])[0]/2), :, :, 0])
        cv2.imwrite('image.png', image)
    
    return res



if __name__ == '__main__':
    
    ## Testing the `mcx_run` function:
    
    # Setting the configuration file
    cfg = {
        'nphoton': 1e7,
        'vol' : np.ones([100, 100, 100], dtype='uint8')
    }
    # Running the funciton
    res = mcx_run(cfg=cfg)
    
    # Acquiring a cube from the function's output
    cube = res['flux'][..., 0]
    
    # Showing the light intensity in the middle of the cube by depth
    plt.subplots()
    plt.plot(cube[50, 50, :])
    plt.tight_layout()
    plt.show()