from matplotlib import patches
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random as random


def display_samples(samples_index,imgs,obs, face_labels, preds_classes=None,preds=None, predictionFlag=False, predictionType="None"):
    """This function randomly displays 20 images with their observed labels 
    and their predicted ones(if preds_classes and preds are provided)"""
    n = 0
    nrows = 2
    ncols = 4
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(8,6))
    plt.subplots_adjust(wspace=0, hspace=0)
    for row in range(nrows):
        for col in range(ncols):
            index = samples_index[n]
            ax[row,col].imshow(imgs[index], cmap='gray')
            ax[row,col].axis('off')
            
            actual_label = face_labels[obs[index]].split("_")[0]
            actual_text = "Actual: {}".format(actual_label)
            
            font0 = FontProperties()
            font = font0.copy()
        
            
            ax[row,col].text(1, 250, actual_text , horizontalalignment='left', fontproperties=font,
                    verticalalignment='top',fontsize=8, color='black')
 
            if predictionFlag:
                predicted_label = face_labels[preds_classes[index]].split('_')[0]
                if predictionType == "RéseauNeurones":
                    predicted_proba = max(preds[index])*100
                    predicted_text = "{} : {:.0f}%".format(predicted_label,predicted_proba)
                else:
                    predicted_text = "Predicted: {}".format(predicted_label)
            
                ax[row,col].text(1, 279, predicted_text , horizontalalignment='left', fontproperties=font,
                    verticalalignment='top',fontsize=8, color='black')
            n += 1

def plot_confusion_matrix(cm, 
                          classes,
                          normalize=False,
                          title='Matrice de confusion',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (7, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Identités réelles', fontweight = "bold")
    plt.xlabel('Identités prédites', fontweight = "bold")

def pick_up_random_element(elem_type,array):
    """This function randomly picks up one element per type in the array"""
    return int(random.choice(np.argwhere(array == elem_type)))