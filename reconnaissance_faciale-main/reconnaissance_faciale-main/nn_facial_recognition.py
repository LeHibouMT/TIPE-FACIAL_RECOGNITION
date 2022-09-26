from neural_network import *
from utility_functions import *
from sklearn.metrics import confusion_matrix
import pandas as pd
import itertools
import matplotlib.pyplot as plt 
import cv2
import numpy as np
import time

# Label index to label name relation
face_labels = {
    0 : 'Colin Powell',
    1 : 'Donald Rumsfeld',
    2 : 'George W Bush',
    3 : 'Gerhard_Schroeder', 
    4 : 'Tony Blair'
 
}

X, y, X_test, y_test = create_dataset('tipe_reconnaissance_faciale/faces_images')
keys = np.array(range(X.shape[0]))

np.random.seed(2)
np.random.shuffle(keys)

split = int(X.shape[0] * 0.80)
X = X[keys]
y = y[keys]

X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

X_train, X_validation = X[0:split], X[split:]
y_train, y_validation = y[0:split], y[split:]

""" print("X_train ", X_train.shape)
print("X_validation ", X_validation.shape)
print("y_train ", y_train.shape)
print("y_validation ", y_validation.shape) """

#display_samples(keys[0:8], X,y)
#plt.show()


# Création d'une instance de la classe Model
model = Model()

# On définit l'architecture du modèle
model.add(Layer_Dense(X_train.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 5))
model.add(Activation_Softmax())

model.set(
    loss = Loss_CategoricalCrossentropy(),
    optimizer = Optimizer_Adam(decay = 1e-3),
    accuracy = Accuracy_Categorical()
)

model.finalize()

#model.load_parameters('tipe_reconnaissance_faciale/neural_network_v2.params')

""" Entraînement du modèle """

start_time = time.time()
model.train(X_train,y_train, validation_data = (X_validation, y_validation), epochs = 15, batch_size=256, print_every=1)
chrono_training = time.time() - start_time
print("\nDurée d'entraînement du réseau de neurones : {:0.2f} secondes".format(chrono_training))

""" Prédiction avec le réseau de neurones """

print("\nPerformance du modèle : ")
model.evaluate(X_test, y_test)
start_time_prediction = time.time()
y_pred = model.predict(X_test)
chrono_prediction = time.time() - start_time_prediction
print("\nTemps d'exécution pour la prédiction avec le réseau de neurones : {:0.2f} secondes ".format(chrono_prediction))
y_pred = np.argmax(y_pred, axis=1)
#print(np.argmax(y_pred, axis=1))

print("Matrice de confusion\n\r", confusion_matrix(y_test, y_pred), "\n")

model.save_parameters('tipe_reconnaissance_faciale/neural_network_v2.params')


# Read an image
#image_data = cv2.imread('tipe_reconnaissance_faciale/faces_images/test/0/Colin_Powell_0174.jpg', cv2.IMREAD_GRAYSCALE)
image_data = cv2.imread('tipe_reconnaissance_faciale/colin_powell_example4.jpeg', cv2.IMREAD_GRAYSCALE)


""" cv2.imshow('Image originale', image_data)
cv2.waitKey(0) 
cv2.destroyAllWindows() """

# Resize to the same size as X.shape * X.shape image
image_data = cv2.resize(image_data, (250, 250))

""" cv2.imshow('Nouvelle image', image_data)
cv2.waitKey(0) 
cv2.destroyAllWindows()  """

# Invert image colors
image_data = 255 - image_data

# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

np.set_printoptions(precision=3, suppress=True)

# Predict on the image
confidences = model.predict(image_data)
print('Probabilités des classes : ', confidences * 100, "\n")
# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

# Get label name from label index
prediction = face_labels[predictions[0]]

print("Résultat de la reconnaissance faciale :",prediction, "\n")


confusion_mtx = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(confusion_mtx, classes = list(face_labels.values()), title="Matrice de confusion (Réseau de neurones)")
plt.show()


#y_pred_classes = np.argmax(y_pred,axis = 0) 
samples = np.random.randint(0,161, size = 8)
confidences = model.predict(X_test)
#print(X_test.reshape((161, 250, 250)).shape)
display_samples(samples, 
                X_test.reshape((161, 250, 250)), 
                obs= y_test, 
                face_labels = face_labels,
                preds =confidences, 
                preds_classes=y_pred, 
                predictionFlag=True,
                predictionType="RéseauNeurones")
plt.show()
#print(model.get_parameters())
