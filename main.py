# import libs
import tensorflow as tf
from keras.models import Sequential, Model
from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import L2
import os
# import matplotlib.pyplot as plt

# Image directory
train_data_dir='data/train/'
validation_data_dir='data/test/'

# Data augmentation
train_data_gen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=30,
									shear_range=0.3,
									zoom_range=0.3,
									horizontal_flip=True,
									fill_mode='nearest')

val_data_gen = ImageDataGenerator(rescale=1./255)


# Preprocessing
train_generator = train_data_gen.flow_from_directory(
					train_data_dir,
					color_mode='rgb',
					target_size=(48, 48),
					batch_size=64,
					class_mode='categorical')

val_generator = val_data_gen.flow_from_directory(
					validation_data_dir,
					color_mode='rgb',
					target_size=(48, 48),
					batch_size=64,
					class_mode='categorical')

class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

img, label = train_generator.__next__() 

# Create model Structure
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Build your model on top of MobileNet
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256,activation='relu',kernel_regularizer=L2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(7,activation='softmax'))  # 7 classes for emotions

# mobilenetV2_model= MobileNetV2(weights='imagenet',
#                                 classes=7,
#                                 include_top=False,
#                                 input_shape=(48,48, 3))

# x= mobilenetV2_model.output
# x= GlobalAveragePooling2D()(x)
# x= Dense(1024,activation='relu')(x)
# x= Dense(512,activation='relu')(x)
# x= BatchNormalization()(x)
# x= Dropout(0.2)(x)

# prediction= Dense(7, activation = 'softmax')(x) #cikti

# model= Model(inputs= mobilenetV2_model.input, outputs= prediction)

# Params
optim = Adam(learning_rate=0.0001, decay=1e-6)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])

print(model.summary())

# Early stopping
# checkpoint =  ModelCheckpoint("model/model_25.weights.h5",monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# early_stopping = EarlyStopping(monitor='val_loss',min_delta=0,patience=3,verbose=1,restore_best_weights=True)
early_stopping = EarlyStopping(monitor='val_loss',   
                               mode='max',   
                               verbose=1,   
                               patience=50,  
                               baseline=0.4,
                               min_delta=0.0001,
                               restore_best_weights=False)

# reduce_lr_rate = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3, verbose=1, min_delta=0.001)

callback_list = [ early_stopping]


train_path = "data/train/"
test_path = "data/test"

# Count of train and test images
num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)
    
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)

print(num_train_imgs) #27809
print(num_test_imgs) #7178
epochs = 60


# Train the model/network
history = model.fit(
    train_generator,
    steps_per_epoch = num_train_imgs//64,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=num_test_imgs//64,
    callbacks=callback_list,
)

# Save model structure in json 
model_json = model.to_json()
with open('model/model_20_new.json', 'w') as json_file:
    json_file.write(model_json)

# Save trained model
model.save_weights('model/model_20_new.weights.h5')

# Plot graph of loss and accuracy
# Assuming you have the training history object named 'history'
# Extract accuracy and loss values from history
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Create the plot
# plt.figure(figsize=(10, 6))

# # Plot accuracy
# plt.subplot(121)  # Create first subplot for accuracy
# plt.plot(train_acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# # Plot loss
# plt.subplot(122)  # Create second subplot for loss
# plt.plot(train_loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# # Improve readability and aesthetics (optional)
# plt.grid(True)
# plt.tight_layout()

# # Display the plot
# plt.show()
