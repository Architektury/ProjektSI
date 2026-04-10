import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape, y_train.shape, X_test.shape, y_test.shape

# Display the first image in the dataset as a data matrix
plt.imshow(X_train[0], cmap="gray")
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()

# Display the values of each pixel in the image
print("Pixel values:")
for row in X_train[0]:
    for pixel in row:
        print("{:3}".format(pixel), end=" ")
    print()

# Display some sample images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0
# Reshape to add channel dimension
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# create an input layer
input_layer = tf.keras.layers.Input(shape=(28, 28, 1)) # 28x28 pixel images with a single color channel

# CNN model building

model = tf.keras.Sequential([
    input_layer, # input layer
    layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu'), # convolutional layer
    # filter is the number of filters we want to apply
    # kernel is the size of window/filter moving over the image
    layers.Conv2D(filters=10, kernel_size=(3, 3),  activation='relu'), # convolutional layer
    layers.MaxPooling2D(), # pooling layer

    layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu'), # convolutional layer
    layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu'), # convolutional layer
    layers.MaxPooling2D(), # pooling layer

    layers.Flatten(), # flatten layer
    layers.Dense(10, activation='softmax') # output layer # why did we add 10?
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=64)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


"""To przerabiasz z klasyfikacji cyfr na klasyfikację autora. W CNN.py obecnie model uczy się na MNIST, więc rozpoznaje 10 cyfr. Dla autorów musisz zmienić 3 rzeczy: dane, etykiety i ostatnią warstwę.

Najprostszy schemat jest taki:

Z każdego pliku autora wycinaj próbki słów albo linijek z obrazów.
Każdej próbce przypisz etykietę autora, np. autor1 = 0, autor2 = 1, itd.
Zamiast wyjścia 10-klasowego daj wyjście o liczbie autorów, u Ciebie najpewniej 8.
W Twoim datasetcie już widać gotową bazę do tego podejścia, bo word_display.py pokazuje, że słowa są opisane w pliku word_places.txt i można je wycinać z obrazów. To jest lepsze niż wrzucanie całych stron, bo model uczy się stylu pisma, a nie treści tekstu.

Technicznie powinieneś:

wczytać obrazy z folderów autorów zamiast MNIST,
przeskalować wszystkie próbki do jednego rozmiaru,
zamienić etykiety na autorów,
ustawić ostatnią warstwę jako Dense(8, activation='softmax'),
użyć sparse_categorical_crossentropy albo categorical_crossentropy,
dzielić dane tak, żeby próbki z jednej strony nie trafiały jednocześnie do treningu i testu.
Jeśli chcesz rozpoznawać autora z całej kartki, a nie z pojedynczego słowa, zwykle robi się to tak:

najpierw model ocenia wiele słów z tej kartki,
potem wyniki się uśrednia albo bierze większość głosów,
dopiero z tego wychodzi autor całego tekstu.
W Twoim aktualnym pliku są też drobne błędy techniczne, które trzeba poprawić przy przeróbce: brakuje importu plt, a w modelu używasz layers.Conv2D, mimo że nie importujesz layers. Jeśli chcesz, mogę Ci od razu przerobić CNN.py na wersję do rozpoznawania autorów z Twojego datasetu."""