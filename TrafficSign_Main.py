# import numpy as np
# import tensorflow
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# from keras.utils.np_utils import to_categorical
# from keras.layers import Dropout, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# import cv2
# from sklearn.model_selection import train_test_split
# import pickle
# import os
# import pandas as pd
# import random
# from keras.preprocessing.image import ImageDataGenerator

# path = "myData"
# labelFile = 'labels.csv'
# batch_size_val = 50
# steps_per_epoch_val = 2000
# epochs_val = 30
# imageDimensions = (32, 32, 3)
# testRatio = 0.2
# validationRatio = 0.2

# count = 0
# images = []
# classNo = []
# myList = os.listdir(path)
# print("Total Classes Detected:", len(myList))
# noOfClasses = len(myList)
# print("Importing Classes.....")
# for x in range(0, len(myList)):
#     myPicList = os.listdir(path + "/" + str(count))
#     for y in myPicList:
#         curImg = cv2.imread(path + "/" + str(count) + "/" + y)
#         images.append(curImg)
#         classNo.append(count)
#     print(count, end=" ")
#     count += 1
# print(" ")
# images = np.array(images)
# classNo = np.array(classNo)

# X_train, X_test, y_train, y_test = train_test_split(
#     images, classNo, test_size=testRatio)
# X_train, X_validation, y_train, y_validation = train_test_split(
#     X_train, y_train, test_size=validationRatio)

# print("Data Shapes")
# print("Train", end="")
# print(X_train.shape, y_train.shape)
# print("Validation", end="")
# print(X_validation.shape, y_validation.shape)
# print("Test", end="")
# print(X_test.shape, y_test.shape)
# assert (X_train.shape[0] == y_train.shape[0]), "The number of images in not equal to the number of labels in training set"
# assert (X_validation.shape[0] == y_validation.shape[0]), "The number of images in not equal to the number of labels in validation set"
# assert (X_test.shape[0] == y_test.shape[0]), "The number of images in not equal to the number of labels in test set"
# assert (X_train.shape[1:] == (imageDimensions)), " The dimensions of the Training images are wrong "
# assert (X_validation.shape[1:] == (imageDimensions)), " The dimensions of the Validation images are wrong "
# assert (X_test.shape[1:] == (imageDimensions)), " The dimensions of the Test images are wrong"

# data = pd.read_csv(labelFile)
# print("data shape ", data.shape, type(data))

# num_of_samples = []
# cols = 5
# num_classes = noOfClasses
# fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
# fig.tight_layout()
# for i in range(cols):
#     for j, row in data.iterrows():
#         x_selected = X_train[y_train == j]
#         axs[j][i].imshow(x_selected[random.randint(
#             0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
#         axs[j][i].axis("off")
#         if i == 2:
#             axs[j][i].set_title(str(j) + "-" + row["Name"])
#             num_of_samples.append(len(x_selected))

# print(num_of_samples)
# plt.figure(figsize=(12, 4))
# plt.bar(range(0, num_classes), num_of_samples)
# plt.title("Distribution of the training dataset")
# plt.xlabel("Class number")
# plt.ylabel("Number of images")
# plt.show()


# def grayscale(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return img


# def equalize(img):
#     img = cv2.equalizeHist(img)
#     return img


# def preprocessing(img):
#     img = grayscale(img)
#     img = equalize(img)
#     img = img / 255
#     return img


# X_train = np.array(list(map(preprocessing, X_train)))
# X_validation = np.array(list(map(preprocessing, X_validation)))
# X_test = np.array(list(map(preprocessing, X_test)))
# cv2.imshow("GrayScale Images",
#            X_train[random.randint(0, len(X_train) - 1)])

# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
# X_validation = X_validation.reshape(
#     X_validation.shape[0], X_validation.shape[1])
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

# dataGen = ImageDataGenerator(width_shift_range=0.1,
#                              height_shift_range=0.1,
#                              zoom_range=0.2,
#                              shear_range=0.1,
#                              rotation_range=10)
# dataGen.fit(X_train)
# batches = dataGen.flow(X_train, y_train,
#                        batch_size=20)
# X_batch, y_batch = next(batches)

# fig, axs = plt.subplots(1, 15, figsize=(20, 5))
# fig.tight_layout()

# for i in range(15):
#     axs[i].imshow(X_batch[i].reshape(imageDimensions[0], imageDimensions[1]))
#     axs[i].axis('off')
# plt.show()

# y_train = to_categorical(y_train, noOfClasses)
# y_validation = to_categorical(y_validation, noOfClasses)
# y_test = to_categorical(y_test, noOfClasses)


# def myModel():
#     no_Of_Filters = 60
#     size_of_Filter = (5, 5)
#     size_of_Filter2 = (3, 3)
#     size_of_pool = (2, 2)
#     no_Of_Nodes = 500
#     models = Sequential()
#     models.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimensions[0], imageDimensions[1], 1),
#                        activation='relu')))
#     models.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
#     models.add(MaxPooling2D(pool_size=size_of_pool))

#     models.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
#     models.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
#     models.add(MaxPooling2D(pool_size=size_of_pool))
#     models.add(Dropout(0.5))

#     models.add(Flatten())
#     models.add(Dense(no_Of_Nodes, activation='relu'))
#     # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
#     models.add(Dropout(0.5))
#     models.add(Dense(noOfClasses, activation='softmax'))  # OUTPUT LAYER
#     # COMPILE MODEL
#     models.compile(
#         Adam(lr=0.001), loss='categorical_cross entropy', metrics=['accuracy'])
#     return models


# model = myModel()
# print(model.summary())
# history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
#                               steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
#                               validation_data=(X_validation, y_validation), shuffle=1)

# plt.figure(1)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['training', 'validation'])
# plt.title('loss')
# plt.xlabel('epoch')
# plt.figure(2)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.legend(['training', 'validation'])
# plt.title('Accuracy')
# plt.xlabel('epoch')
# plt.show()
# score = model.evaluate(X_test, y_test, verbose=0)
# print('Test Score:', score[0])
# print('Test Accuracy:', score[1])

# # STORE THE MODEL AS A PICKLE OBJECT
# pickle_out = open("model_trained.p", "wb")  # wb = WRITE BYTE
# pickle.dump(model, pickle_out)
# pickle_out.close()
# cv2.waitKey(0)


from keras.models import load_model
from sklearn.metrics import accuracy_score
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
os.chdir('F:/Traffic Sign Recognition')

data = []
labels = []
classes = 43
cur_path = os.getcwd()

cur_path

for i in range(classes):
    path = os.path.join(cur_path, 'myData', str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '\\' + a)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(e)

data = np.array(data)
labels = np.array(labels)

np.save('./training/data', data)
np.save('./training/target', labels)

data = np.load('./training/data.npy')
labels = np.load('./training/target.npy')

print(data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5),
          activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
# We have 43 classes that's why we have defined 43 in the dense
model.add(Dense(43, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


epochs = 20
history = model.fit(X_train, y_train, batch_size=32,
                    epochs=epochs, validation_data=(X_test, y_test))

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


def testing(testcsv):
    y_test = pd.read_csv(testcsv)
    label = y_test["ClassId"].values
    imgs = y_test["Path"].values
    data=[]
    for img in imgs:
        image = Image.open(img)
        image = image.resize((30,30))
        data.append(np.array(image))
    X_test=np.array(data)
    return X_test,label


X_test, label = testing('Test.csv')

Y_pred = model.predict_classes(X_test)
Y_pred

print(accuracy_score(label, Y_pred))

model.save("./training/TSR.h5")

os.chdir(r'D:\Traffic_Sign_Recognition')
model = load_model('./training/TSR.h5')

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def test_on_img(img):
    data=[]
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = model.predict_classes(X_test)
    return image,Y_pred
plot,prediction = test_on_img(r'D:\Traffic_Sign_Recognition\Test\00500.png')
s = [str(i) for i in prediction] 
a = int("".join(s)) 
print("Predicted traffic sign is: ", classes[a])
plt.imshow(plot)
plt.show()