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


def SuperLearning(models, train_dir, predict_dir, classes, **kwargs):
    # Names of train_names have to be equal to names in label_names (img by img)
    # It does not classify images with label set to zero/0 (background)
    # Function leaves a background folder with all label images
    # 'scale_images' is to scale output/classified images
    
    # Getting all the 'keyword arguments' entered by the user
    save_images = kwargs.get('save_images')
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
        print('\n\n\n- - - - - - - - - - - - -\n\n\n')
    
    return results


if __name__ == '__main__':
    train_dir = r'C:\Users\marlo\iCloudDrive\Downloads\Deletar Posteriormente\Smaller Images JPG\train'
    predict_dir = r'C:\Users\marlo\iCloudDrive\Downloads\Deletar Posteriormente\Smaller Images JPG\test'
    label_names = ['healthy', 'damaged', 'white']
    models = ['Random Forest']
    classes = 3
    
    results = SuperLearning(
        models, train_dir, predict_dir, classes,label_names=label_names,
        show_images=True, scale_images=True, save_images=True
    )

