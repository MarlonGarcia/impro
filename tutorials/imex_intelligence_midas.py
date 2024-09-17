'''
The `midas` class is used for depth estimation using the MiDaS model.
See MiDaS repository for more information: https://github.com/isl-org/MiDaS
This code was based on MiDaS hub in PyTorch, that can be found here:
https://pytorch.org/hub/intelisl_midas_v2/

Args:
    model_type (str): The type of MiDaS model to use.
    dir_input (str): The directory path of the input images.
    dir_output (str): The directory path to save the output images.
    **kwargs: Additional keyword arguments.
        show (bool): Whether to show the images. Default is False.
        save (bool): Whether to save the images. Default is False.

Observations: Please, do not use Matplotlib to print the image returned from this
library. It can cause some problems.
'''

import impro



# Choose a directory with the images to predicth depth
dir_input = r'C:\Users\marlo\Downloads\MiDas\2024.05.24 - Testes Iniciais\2d images'
# Choose a directory to save predicted images
dir_output = r'C:\Users\marlo\Downloads\MiDas\2024.05.24 - Testes Iniciais\depth images'

# Choosing the model type (e.g. 'DPT_BEiT_L_512', 'DPT_Large', 'MiDaS_small')
model_type = 'DPT_BEiT_L_512'

# Define input parameters
model = impro.midas(model_type, dir_input, dir_output, show=True, save=True)
# Execute MiDaS for prediction
_ = model.predict()
