from impro import intelligence


# Enter here the directory of your training dataset
train_dir = r'C:\Users\marlo\My Drive\College\Biophotonics Lab\Research\Data\RGB Images\01) 25.06.20 - Erika\imagens'

# Enter here the directory of the data to be predicted
predict_dir = r'C:\Users\marlo\My Drive\College\Biophotonics Lab\Research\Data\RGB Images\01) 25.06.20 - Erika\PDT_50mW_28_11_19_14B'

# Naming the three classes to be predicted
label_names = ['healthy', 'damaged', 'white']

# Name of the models to be used
models = ['Random Forest']

# Number of classes
classes = 3


# Running the function
intelligence.SuperLearning(models, train_dir, predict_dir, classes,
                           label_names=label_names, show_images=True,
                           scale_images=False)

