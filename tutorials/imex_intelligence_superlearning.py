from impro import intelligence


# Enter here the directory of your training dataset
train_dir = r'C:\Users\marlo\iCloudDrive\Downloads\Deletar Posteriormente\Smaller Images JPG\train'
# train_dir = r'C:\Users\marlon.garcia\Downloads\Deletar Futuramente\Smaller Images JPG\train'

# Enter here the directory of the data to be predicted
predict_dir = r'C:\Users\marlo\iCloudDrive\Downloads\Deletar Posteriormente\Smaller Images JPG\test'
# predict_dir = r'C:\Users\marlon.garcia\Downloads\Deletar Futuramente\Smaller Images JPG\test'

# Naming the three classes to be predicted
label_names = ['healthy', 'damaged', 'white']

# Name of the models to be used
models = ['Random Forest']

# Number of classes
classes = 3


# Running the function
results = intelligence.SuperLearning(
    models, train_dir, predict_dir, classes,label_names=label_names,
    show_images=True, scale_images=True, save_images=True
)

