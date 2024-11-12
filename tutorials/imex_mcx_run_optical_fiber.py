# -*- coding: utf-8 -*-
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

import numpy as np
import matplotlib.pyplot as plt
from impro import monte_carlo as mc


## Testing the `mcx_run` function:
# Setting the configuration file
cfg = {
    'nphoton': 1e7,
    'vol' : np.ones([100, 100, 100], dtype='uint8'),
}

# Running the funciton
res = mc.mcx_run(cfg=cfg)

# Acquiring a cube from the function's output
cube = res['flux'][..., 0]

# Showing the light intensity in the middle of the cube by depth
plt.subplots()
plt.plot(cube[30, 30, :])
plt.tight_layout()
plt.show()