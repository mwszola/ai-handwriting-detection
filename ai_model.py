import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Odczyt danych

data = pd.read_csv(r"/Users/matthew/Desktop/archive/A_Z Handwritten Data.csv").astype('float32')

# Podział danych na zdjęcia oraz ich opisy
X = data.drop('0', axis=1)
y = data['0']

# Przygotowanie danych, aby miały odpowiedni format
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_x = np.reshape(train_x.values, (train_x.shape[0], 28, 28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28, 28))

# Skojarzenie wartości numerycznych z literami alfabetu
alfabet = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
           13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
           25: 'Z'}

y_int = np.int0(y)
count = np.zeros(26, dtype='int')
for i in y_int:
    count[i] += 1

alphabets = []
for i in alfabet.values():
    alphabets.append(i)

# Utworzenie wykresu prezentującego zbiór danych, na podstawie których będzie trenowany i testowany model
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.barh(alphabets, count)

plt.xlabel("Number of elements ")
plt.ylabel("Alphabets")
plt.grid()
plt.show()

# Utworzenie losowej próbki danych
shuff = shuffle(train_x[:100])

fig, ax = plt.subplots(3, 3, figsize=(10, 10))
axes = ax.flatten()

for i in range(9):
    _, shu = cv2.threshold(shuff[i], 30, 200, cv2.THRESH_BINARY)
    axes[i].imshow(np.reshape(shuff[i], (28, 28)), cmap="Greys")
plt.show()


# Przygotowanie danych, aby byly prawidlowo interperetowane przez model

train_X = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

# Zmiana struktury zestawu zdjęć do trenowania oraz testowania w taki sposób, że będą mogły zostać umieszczone w modelu
train_yOHE = to_categorical(train_y, num_classes=26, dtype='int')
test_yOHE = to_categorical(test_y, num_classes=26, dtype='int')

# CNN Model - model na podstawie próbki do trenowania
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(26, activation="softmax"))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_X, train_yOHE, epochs=1, validation_data=(test_X, test_yOHE))

# Podsumowanie, ktore daje nam informacje o tym, gdzie rozne warstwy sa zdefiniowane w modelu
model.summary()
model.save(r'model.h5')

print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])

# Wykonywanie przeiwdywan na danych testowych
fig, axes = plt.subplots(3, 3, figsize=(8, 9))
axes = axes.flatten()

for i, ax in enumerate(axes):
    img = np.reshape(test_X[i], (28, 28))
    ax.imshow(img, cmap="Greys")

    pred = alfabet[np.argmax(test_yOHE[i])]
    ax.set_title("Prediction: " + pred)
    ax.grid()


# Funkcja pomocnicza do przetworzenia obrazu
def convertImage(filename):
    # Wykonywanie przewidywan na zdjeciu z zewnatrz
    name = filename
    img = cv2.imread(fr'/Users/matthew/Desktop/letters/{name}')
    img_copy = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 440))

    img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    img_final = cv2.resize(img_thresh, (28, 28))
    img_final = np.reshape(img_final, (1, 28, 28, 1))

    img_pred = alfabet[np.argmax(model.predict(img_final))]

    cv2.putText(img, "Prediction: " + img_pred, (20, 410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color=(255, 0, 30))
    cv2.imshow('handwritten character recognition _ _ _ ', img)


convertImage('letter_a.png')

while 1:
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
