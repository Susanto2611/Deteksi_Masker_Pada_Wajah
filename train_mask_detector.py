#Source Code Model Training
# mengimport paket
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 2
BS = 32
#Menimport Dataset dan kategori didalamnya 
DIRECTORY = r"D:\Download\Sus\Face_Mask_Detection-master\Face_Mask_Detection-master\dataset" 
CATEGORIES = ["with_mask", "without_mask"]
print("[INFO] loading images...")

data = [] #insialisasi list array yang akan dipakai untuk training
labels = [] ##insialisasi list array  yang akan dipakai untuk label
#Membaca  gambar di dataset
for category in CATEGORIES: 
    path = os.path.join(DIRECTORY, category) #membaca direktori dataset untuk 'with mask' dan 'without mask'
    for img in os.listdir(path):
    	img_path = os.path.join(path, img) #membaca gambar pada masing2 direktori
    	image = load_img(img_path, target_size=(224, 224)) #merubah ukuran gambar ke  224x224
    	image = img_to_array(image) #mengekstraksi array dari gambar
    	image = preprocess_input(image) #proses gambar
    	data.append(image) #memasukkan array image kedalam list data untuk dataset training
    	labels.append(category) #memasukkan array image kedalam list data untuk pelabelan

# perform one-hot encoding on the labels : merubah ke categorical data
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32") #merubah tipe data menjadi float 32
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42) #split dataset menjadi dataset training dan testing
													  #dataset training 80 %, testing 20% menggunakan stratify sampling

# construct the training image generator for data augmentation : preprocessing image
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"]) #mengcompile model

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS), #trainX : data training; trainY: data validasi
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS) #training model

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS) #evaluasi model

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_)) #mencetak report dari hasil training

#menyimpan model hasil training
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5") 

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")