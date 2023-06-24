import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os, glob
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

data_dir = (r"D:/DL/Plant Nutrient Deficiencies/CODE/Plants")
train_dir = os.path.join(data_dir)
Plants = ['Calcium deficiency','Complete nutrition', 'Iron Deficiency', 'Magnesium Deficiencies', 'Nitrogen Deficiency', 'Potassium Deficiency']
print(Plants)

train_data = []
for defects_id, sp in enumerate(Plants):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train_data.append(['{}/{}'.format(sp, file), defects_id, sp])

train = pd.DataFrame(train_data, columns=['File', 'LabelID', 'Label'])
train.head()

SEED = 42
train = train.sample(frac=1, random_state=SEED)
train.index = np.arange(len(train))  # Reset indices
train.tail()

def plot_defects(defect_types, rows, cols):
    fig, ax = plt.subplots(rows, cols, figsize=(8, 8))
    defect_files = train['File'][train['Label'] == defect_types].values
    n = 0
    for i in range(rows):
        for j in range(cols):
            image_path = os.path.join(data_dir, defect_files[n])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].imshow(cv2.imread(image_path))
            n += 1


plot_defects('Calcium deficiency', 3, 3)

def plot_defects(defect_types, rows, cols):
    fig, ax = plt.subplots(rows, cols, figsize=(8, 8))
    defect_files = train['File'][train['Label'] == defect_types].values
    n = 0
    for i in range(rows):
        for j in range(cols):
            image_path = os.path.join(data_dir, defect_files[n])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].imshow(cv2.imread(image_path))
            n += 1


plot_defects('Iron Deficiency', 3, 3)

training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = training_datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    class_mode='categorical')

IMAGE_SIZE = 128


def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath))


def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)


x_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(train['File'].values)):
    image = read_image(file)
    if image is not None:
        x_train[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))

        x_train = x_train / 255.

        num_classes = 10
        y_train = train['LabelID'].values
        y_train = to_categorical(y_train, num_classes)

        X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=SEED)
        print('Train Shape: {}'.format(X_train.shape))

fig, ax = plt.subplots(1, 5, figsize=(15, 15))
for i in range(5):
    ax[i].set_axis_off()
    ax[i].imshow(X_train[i])
    ax[i].set_title(Plants[np.argmax(Y_train[i])])


from tensorflow import keras
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [128, 128,3]),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(32, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

model.summary()

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
hist=model.fit(X_train,Y_train,epochs=15,batch_size=10,validation_data=(X_test,Y_test))


model.evaluate(X_test, Y_test)
model.save("D:\DL\Plant Nutrient Deficiencies\CODE\models\ANN.h5")

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plt.save("D:\DL\Plant Nutrient\ANN_acc.png")


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plt.save("D:\DL\Plant Nutrient\ANN_acc.png")

model1 = Sequential()

model1.add(Conv2D(128, kernel_size=(3,3),activation='relu',input_shape=(128,128,3)))
model1.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
model1.add(Conv2D(32, kernel_size=(3,3),activation='relu'))
model1.add(Flatten())
model1.add(Dense(10, activation='softmax'))

model1.summary()

model1.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
hist1=model1.fit(X_train,Y_train,epochs=10,batch_size=10,validation_data=(X_test,Y_test))

model1.evaluate(X_test, Y_test)
model1.save("D:\DL\Plant Nutrient Deficiencies\CODE\models\CNN.h5")

plt.plot(hist1.history['accuracy'])
plt.plot(hist1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plt.save("D:\DL\Plant Nutrient\CNN_acc.png")


plt.plot(hist1.history['loss'])
plt.plot(hist1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plt.save("D:\DL\Plant Nutrient\CNN_loss.png")

base_model2 = tf.keras.applications.DenseNet121(input_shape=(128, 128, 3), include_top=False,
                          weights='imagenet')
model2 = Sequential()
model2.add(base_model2)
model2.add(GlobalAveragePooling2D())
model2.add(Dense(64, activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.2))
model2.add(Dense(10, activation='sigmoid'))
model2.summary()

model2.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
hist2=model2.fit(X_train,Y_train,epochs=10,batch_size=10,validation_data=(X_test,Y_test))

model2.evaluate(X_test, Y_test)
model2.save("D:\DL\Plant Nutrient Deficiencies\CODE\models\DenseNet121.h5")

plt.plot(hist2.history['accuracy'])
plt.plot(hist2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plt.save("D:\DL\Plant Nutrient\DenseNet121_acc.png")


plt.plot(hist2.history['loss'])
plt.plot(hist2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plt.save("D:\DL\Plant Nutrient\DenseNet121_loss.png")

Accuracy = []

for i in [hist, hist1, hist2]:
    x = i.history.get('accuracy')[-1]
    Accuracy.append(x)

Accuracy

from skimage import io
from keras.preprocessing import image

img = image.load_img(r'D:\DL\Plant Nutrient Deficiencies\CODE\Plants\Magnesium Deficiencies\gfgjfj.jfif',
                     grayscale=False, target_size=(128, 128))
show_img = image.load_img(r'D:\DL\Plant Nutrient Deficiencies\CODE\Plants\Magnesium Deficiencies\gfgjfj.jfif',
                          grayscale=False, target_size=(200, 200))
Plants = ['Calcium deficiency', 'Complete nutrition', 'Iron Deficiency', 'Magnesium Deficiencies',
          'Nitrogen Deficiency', 'Potassium Deficiency']
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

x /= 255

custom = model.predict(x)
print(custom[0])

plt.imshow(show_img)
plt.show()

a = custom[0]
ind = np.argmax(a)

print('Prediction:', Plants[ind])