# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import imfun
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch
import pillow_heif


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



def SuperLearning(models, train_dir, predict_dir, classes, **kwargs):
    # Names of train_names have to be equal to names in label_names (img by img)
    # It does not classify images with label set to zero/0 (background)
    # Function leaves a background folder with all label images
    # 'scale_images' is to scale output/classified images
    
    # Getting all the 'keyword arguments' entered by the user
    save_images = kwargs.get('save_images', True)
    save_results = kwargs.get('save_results', True)
    save_model = kwargs.get('save_model')
    trees = kwargs.get('trees', 100)
    label_dir = kwargs.get('label_dir')
    label_names = kwargs.get('label_names')
    show_images = kwargs.get('show_images')
    scale_images = kwargs.get('scale_images', True)
    cmap = kwargs.get('cmap')
    
    # Converting to lowercase all letters in 'models'
    models = [item.lower() for item in models]
    
    # Labeling images, if the directory for label images was not passed 
    if label_dir is None:
        temp = imfun.im2label(train_dir, classes, scale = False,
                              label_names = label_names)
    del temp
    
    # Eliminating non-image-files (important for folders from online drives)
    train_names = imfun.list_images(train_dir)
    
    train_data = []
    train_class = []
    for name in train_names:
        train_image = cv2.imread(os.path.join(train_dir, name))
        path = os.path.join(os.path.dirname(train_dir),
                            os.path.basename(train_dir) + ' labels')
        train_label = cv2.imread(os.path.join(path, name))
        # Adding the pixels' data and labels to `train_data` and `train_class`
        for label in range(1, classes+1):
            # Adding just the pixels where `train_label` has values = `label`
            train_data.append(train_image[train_label[...,0] == label])
            # Creates an array full of values `label`, with shape of data above
            train_class.append(np.full(train_data[-1].shape[0], label))
    # Concatenating the intire list of different arrays in a long, thin arrays
    train_data = np.concatenate(train_data, axis = 0)
    train_class = np.concatenate(train_class, axis = 0)
    # Shuffling the data (important for model training)
    train_data, train_class = shuffle(train_data, train_class, random_state=42)
    
    # Starting the results dictionary
    results = {}
    
    # Random Forest Model
    if 'random forest' in models:
        print('\n\n\n- - - Random Forest - - -\n')
        print('\nTraining Model:\n\n')
        class ProgressRandomForest(RandomForestClassifier):
            def fit(self, X, y):
                for i in tqdm(range(self.n_estimators)):
                    super().fit(X, y)
                return self
        
        clf = ProgressRandomForest(n_estimators=trees)
        # Effectively training
        clf.fit(train_data, train_class)
        # Deliting unused variables
        del train_data, train_class
        
        # Extracting image names from `predict_dir`
        predict_names = imfun.list_images(predict_dir)
        
        
        # Saving model, if required
        if save_model:
            print('\n\nSaving Model...\n\n')
            os.chdir(predict_dir)
            try:
                os.mkdir('trained models')
            except:
                pass
            os.chdir('trained models')
            with open('random_forest_model.pkl', 'wb') as file:
                pickle.dump(clf, file)
        
        # Creating the directory to save predicted images
        os.chdir(predict_dir)
        try:
            os.mkdir('predicted images')
        except:
            pass
        os.chdir('predicted images')
        
        # Predicting with Random Forest
        print('\n\nPredicting with Random Forest:\n\n')
        for name in tqdm(predict_names):
            pred_image = cv2.imread(os.path.join(predict_dir, name))
            if pred_image is None:
                raise ValueError(f'Error while reading image {name}')
            pred_data = imfun.im2flat(pred_image)
            # Actually predicting
            out_data = clf.predict(pred_data)
            # Calculating the image back from the flat vector
            out_image = imfun.flat2im(out_data, pred_image.shape)
            out_image = np.array(out_image, dtype='uint8')
            if scale_images:
                out_image = imfun.scale255(out_image)
            if cmap:
                out_image = cv2.applyColorMap(out_image, cmap)
            if save_images:
                cv2.imwrite(name, out_image)
            if show_images:
                plt.subplots()
                if cmap:
                    out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
                plt.imshow(out_image)
                plt.title(f'Predicted Image for {name}')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
        
        # Calculating the metrics
        print('\n\nCalculating Model Metrics:\n\n')
        # `im2label` will store the label images at `predict_dir +' labels'`
        temp = imfun.im2label(predict_dir, classes, scale = False,
                              label_names=label_names, save_images=True)
        del temp
        
        # After storing the images, we will extract and prepare the data
        test_data = []
        test_class = []
        for name in predict_names:
            # Reading image in `predict_dir`
            test_image = cv2.imread(os.path.join(predict_dir, name))
            path = os.path.join(os.path.dirname(predict_dir),
                                os.path.basename(predict_dir) + ' labels')
            # Reading image in `predict_dir+' labels'`
            test_label = cv2.imread(os.path.join(path, name))
            # Extracting data from each class to the data to be tested
            for label in range(1, classes+1):
                # This will compare with label and store in a long, thin data
                test_data.append(test_image[test_label[...,0] == label])
                test_class.append(np.full(test_data[-1].shape[0], label))
        # Concatenating each label of each image in a even longer, thinner data
        test_data = np.concatenate(test_data, axis = 0)
        test_class = np.concatenate(test_class, axis = 0)
        # Predicting the test dataset
        test_out = clf.predict(test_data)
        
        # Classification Report
        report = classification_report(test_class, test_out)
        print(f'\n\nClassification Report:\n{report}\n')
        # Confusion Matrix
        cm = confusion_matrix(test_class, test_out)
        print(f'\nConfusion Matrix:\n{cm}')
        # Adding the results in a tupple
        results['Random Forest'] = (report, cm)
        # Saving results, if required
        if save_results:
            print('\n\nSaving Results...\n\n')
            os.chdir(predict_dir)
            try:
                os.mkdir('results')
            except:
                pass
            os.chdir('results')
            # Salvando o relatório de classificação e a matriz de confusão em um único arquivo TXT
            with open('results_random_forest.txt', 'w') as f:
                # Escrevendo o relatório de classificação
                f.write("Relatório de Classificação:\n")
                f.write(report)
                
                # Adicionando algumas linhas em branco
                f.write("\n\n\n")
                
                # Escrevendo a matriz de confusão
                f.write("Matriz de Confusão:\n")
                df_cm = pd.DataFrame(cm)
                df_cm.to_csv(f, index=False, sep='\t')
        print('\n\n\n- - - - - - - - - - - - -\n\n\n')
    
    return results



def EasySegment():
    ...



def EasyClassify():
    ...


'''
Adicionar aqui:
    - imroiprop
    - roi_stats_in_detph
'''


if __name__ == '__main__':
    train_dir = r'C:\Users\marlon.garcia\Downloads\Deletar Futuramente\Smaller Images JPG2\train'
    train_dir = r'C:\Users\marlo\iCloudDrive\Downloads\Deletar Posteriormente\Smaller Images JPG\train'
    predict_dir = r'C:\Users\marlon.garcia\Downloads\Deletar Futuramente\Smaller Images JPG2\test'
    predict_dir = r'C:\Users\marlo\iCloudDrive\Downloads\Deletar Posteriormente\Smaller Images JPG\test'
    label_names = ['non-healthy', 'healthy', 'white']
    models = ['Random Forest']
    classes = 3
    
    results = SuperLearning(
        models, train_dir, predict_dir, classes,label_names=label_names,
        show_images=True, scale_images=True, save_images=True
    )

