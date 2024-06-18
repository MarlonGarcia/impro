import torch
import cv2
import os
import imfun
import pillow_heif
import numpy as np

class midas():
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

    def __init__(self, model_type, dir_input, dir_output, **kwargs):
        # Getting 'show' variable:
        self.show = kwargs.get('show', False)
        # Getting 'save' variable:
        self.save = kwargs.get('save', False)
        self.model_type = model_type
        self.dir_input = dir_input
        self.dir_output = dir_output
    
    def predict(self, esc=False):
            '''
            Predicts the depth of input images using the MiDaS model.

            Returns:
                output_images (list): List of depth images.
            '''
            
            # Downloading it
            midas = torch.hub.load('intel-isl/MiDaS', self.model_type, trust_repo=True)
            # Setting device
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            print(f'\n\nProcessing on {device}\n\n')
            
            # Casting model to device
            midas.to(device)
            # Setting model in evaluation mode
            midas.eval()
            
            # Loading and executing the correct transformations
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            if self.model_type == 'DPT_BEiT_L_512':
                transform = midas_transforms.beit512_transform
            elif self.model_type == 'DPT_Large' or self.model_type == 'DPT_Hybrid':
                transform = midas_transforms.dpt_transform
            else:
                transform = midas_transforms.small_transform
            
            output_images = []
            for name in os.listdir(self.dir_input):
                # Verify if image is in ".heic" format to import with pillow
                if name[-5:] in ['.HEIC', '.heic']:
                    heif_file = pillow_heif.open_heif(
                        os.path.join(self.dir_input, name),
                        convert_hdr_to_8bit=False
                    )
                    img = np.asarray(heif_file)
                else:
                    # Reading image with OpenCV
                    img = cv2.imread(os.path.join(self.dir_input, name))
                    # Transforming from BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Applying loaded transformation
                input_batch = transform(img).to(device)
                
                # Predicting
                with torch.no_grad():
                    prediction = midas(input_batch)
                    
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode='bicubic',
                        align_corners=False,
                    ).squeeze()
                
                # Casting to 'cpu' to be able to use it and transforming to NumPy
                output = prediction.cpu().numpy()
                # Rescaling to 0-255 range and transforming in 'uint8' variable
                output = imfun.scale255(output)
                
                # If 'show=True' showing all images
                if self.show:
                    cv2.namedWindow('Depth Estimation', cv2.WINDOW_NORMAL)
                    cv2.imshow('Depth Estimation', output)
                    if esc:
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                # If 'save=True' save all images
                if self.save:
                    # If image in '.heic' format, save in '.jpg' format
                    if name[-5:] in ['.HEIC', '.heic']:
                        file_name = os.path.join(self.dir_output,name[0:-5]+'.jpg')
                    else:
                        file_name = os.path.join(self.dir_output, name)
                    print(file_name)
                    cv2.imwrite(file_name, output)
                output_images.append(output)
            
            return output_images



if __name__ == '__main__':
    # Choose a directory with the images to predicth depth
    dir_input = r'C:\Users\marlo\Downloads\MiDas\2024.05.24 - Testes Iniciais\2d images'
    # Choose a directory to save predicted images
    dir_output = r'C:\Users\marlo\Downloads\MiDas\2024.05.24 - Testes Iniciais\depth images'
    
    # Choosing the model type (e.g. 'DPT_BEiT_L_512', 'DPT_Large', 'MiDaS_small')
    model_type = 'DPT_BEiT_L_512'
    
    # Define input parameters
    model = midas(model_type, dir_input, dir_output, show=True, save=True)
    # Execute MiDaS for prediction
    _ = model.predict()
