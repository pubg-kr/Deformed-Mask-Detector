from tensorflow import keras
from tensorflow.keras import datasets, models, layers, utils
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) \
            = datasets.cifar10.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(y_train[0:20])

fig, axs = plt.subplots(2, 5, figsize=(10, 5))
for i in range(2):
  for j in range(5):
    axs[i, j].imshow(x_train[i*5+j])
plt.show()

classes = ('airplane','automobile','bird',
           'cat','deer','dog','frog', 'horse', 'ship','truck')

for y in y_train[0:10]:
  print(classes[y[0]])

print(np.unique(y_train, return_counts=True))


x_train, x_test = x_train/255, x_test/255


model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), padding='same', 
                        activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2), strides=2))
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2,2), strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics='accuracy')


checkpoint = keras.callbacks.ModelCheckpoint("best-model.h5")
early_stopping = keras.callbacks.EarlyStopping(patience=2, 
                                               restore_best_weights=True)
history=model.fit(x_train, y_train, epochs=20, 
          validation_split=0.2, callbacks=[checkpoint, early_stopping])

def plot_history(history):
  plt.figure(figsize=(10, 3))
  plt.subplot(1,2,1)
  plt.plot(history['accuracy'])
  plt.plot(history['val_accuracy'])
  plt.legend(['train', 'val'])

  plt.subplot(1,2,2)
  plt.plot(history['loss'])
  plt.plot(history['val_loss'])
  plt.legend(['train', 'val'])
  plt.show()

plot_history(history.history)


model.evaluate(x_test, y_test)


pred = model.predict(x_test[0:10])
print(pred)

pred = np.argmax(pred, axis=1)
print(pred)

print(y_test[0:10].reshape(10,))

for i in pred:
  print(classes[i], end=',\t')
print()
for i in y_test[0:10].reshape(10,):
  print(classes[i], end=',\t')

fig, axs = plt.subplots(2, 5, figsize=(10, 5))
for i in range(2):
  for j in range(5):
    axs[i, j].imshow(x_test[i*5+j])
plt.show()

