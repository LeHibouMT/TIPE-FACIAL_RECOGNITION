# -*- coding: utf-8 -*-
""" 
- eigenfaces.py
- faces_images : dossier conteant les photos de références
- faces_images/train : dossier contenant les images d'entraînement (pour le réseau de neurones uniquement) - 979 images 
- faces_images/test : dossier contenant les images de test -> on évalue les performances sur ces images (pour le réseau de neurones et eigenfaces) - 161 images
- eingenfaces : Contiendra les eigenfaces

"""

from neural_network import *
from utility_functions import *
from sklearn.metrics import confusion_matrix
from scipy import linalg

import itertools
import matplotlib.pyplot as plt 
import cv2
import numpy as np
import time

#np.random.seed(24)

# Noms des différents individus - Notation liée aux dossiers 
face_labels = {
    0 : 'Colin Powell',
    1 : 'Donald Rumsfeld',
    2 : 'George W Bush',
    3 : 'Gerhard_Schroeder', 
    4 : 'Tony Blair'
 
}

heigh = 250
length= 250

X_train, y_train, X_test, y_test = create_dataset('tipe_reconnaissance_faciale/faces_images')
keys = np.array(range(X_train.shape[0]))
np.random.shuffle(keys)
X_train = X_train[keys]
y_train = y_train[keys]

X_train = (X_train.reshape(X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5


""" Methode Eigenfaces """

print("\nReconnaissance avec la méthode Eigenfaces")

# Calcul, affichage et enregistrement de la moyenne des images 
# moyenne : vecteur de l'image moyenne   
moyenne = np.mean(X_test, 0)
phi = X_test - moyenne
# La decomposition en valeurs singulières nous permet de calculer la matrice d'Eigenfaces (qui comporte les visages propres)
# Sigma et V ne sont pas utiles mais il faut bien les affecter car la fonction linagl.svd() retourne 3 sorties (U, Sigma, V.T)
eigenfaces, sigma, v = linalg.svd(phi.transpose(), full_matrices=False)
weights = np.dot(phi, eigenfaces)


def eigenfaces_predictions(X, y):

    # Liste de sortie contenant les prédictions de la méthode Eigenfaces
    predictions = []
    distances = []
    # Soit X un ensemble de données contenant les pixels tel que:
        # Chaque ligne correspond à une image sous la forme d'un vecteur de taille (1, Nombre de pixels pixels) 

    # Pour toutes les images dans les données, on applique la méthode Eigenfaces
    for i in range(X.shape[0]):
        # Centrage des visages aplatis (on passe de [0, 255] à [-1, 1]) 
        image_test = X[i]

        # On retire à chaque visage, le visage moyen
        # C'est le visage propre ??
        phi2 = image_test - moyenne


        # On calcule les coordonnées du visage dans l'ensemble des visages propres (Eigenfaces)
        w2 = np.dot(phi2,eigenfaces)

        # On compare le visage projeté sur l'ensemble des visages propres à chaque visage propre ?? 
        dist = np.min((weights-w2)**2,axis=1)
    
        # On sélectionne l'indice de du visage le plus proche -> prédiction de la reconnaissance faciale
        indiceImg = np.argmin(dist)
        
        #mindist = np.sqrt(dist[indiceImg])

        predictions.append(y[indiceImg])

    return np.array(predictions)



start_time = time.time()
preds = eigenfaces_predictions(X_test, y_test)
chrono_eigenfaces = time.time() - start_time
print("\nTemps d'exécution pour la prédiction avec les Eigenfaces {:0.2f} secondes".format(chrono_eigenfaces))


#print("Taille de la sortie prédite", preds.shape)
#print("Taille de la sortie réelle", y_test.shape)

print("\nPrécision obtenue avec les Eigenfaces {:0.2f}%".format(np.mean(y_test == preds)))

confusion_mtx = confusion_matrix(y_test, preds)
print("\nMatrice de confusion obtenue avec les Eigenfaces")
print(confusion_mtx)

plot_confusion_matrix(confusion_mtx, classes = list(face_labels.values()), title = "Matrice de confusion (Eigenfaces)")
plt.show()

# lire l'image à tester 
image_test = cv2.imread('tipe_reconnaissance_faciale/tony_blair_example.jpeg', cv2.IMREAD_GRAYSCALE)
image_test = cv2.resize(image_test, (250, 250))
image_test_not_flatten = image_test.copy()
image_test = (image_test.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

#print(distances)

#y_pred_classes = np.argmax(y_pred,axis = 0) 
samples = np.random.randint(0,161, size = 8)
#print(X_test.reshape((161, 250, 250)).shape)
display_samples(samples, 
                X_test.reshape((161, 250, 250)), 
                obs= y_test,
                face_labels = face_labels,
                preds_classes=preds, 
                predictionFlag=True)
plt.show()

