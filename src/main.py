from os import listdir
from os.path import join, isdir
from scipy import ndimage as im
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import imageio
from sys import stdin
from  sklearn.ensemble import GradientBoostingClassifier
from utils import *
import argparse
from matrixtools import *
from src.utils import openImages

parser = argparse.ArgumentParser(description='Facial recognition system.')
parser.add_argument("--kernel", "-k", help="Uses KPCA", action="store_true",
                    default=False)
parser.add_argument("--faces_directory", help="Path to the directory with the faces.", action="store",
                    default='./../att_faces/')
parser.add_argument("--face_test_directory", help="Path to the directory with the faces to test.", action="store",
                    default='./../att_faces/')
parser.add_argument("--eigenfaces", help="How many eigenfaces are used.", action="store", default=50)
parser.add_argument("--training", help="How many photos used for training out of 10.", action="store",
                    choices=[1,2,3,4,5,6,7,8,9,10], type=int, default=6)
args = parser.parse_args()

mypath = args.faces_directory

#image size
horsize     = 92
versize     = 112
areasize    = horsize*versize

#number of figures
personno    = 40
trnperper   = args.training
tstperper   = 10 - args.training
trnno = personno * trnperper
tstno = personno * tstperper

clf = svm.LinearSVC()
#clf = GradientBoostingClassifier()
# TRAINING

images_training, person_training, names_dictionary = openImages(path=mypath, personno=personno, trnperper=trnperper, areasize=areasize)
if args.kernel:
    images_training *= 255.0
    images_training -= 127.5
    images_training /= 127.5

    # KERNEL: polinomial de grado degree
    degree = 2
    K = (np.dot(images_training, images_training.T) / trnno + 1) ** degree
    # esta transformación es equivalente a centrar las imágenes originales...
    unoM = np.ones([trnno, trnno]) / trnno
    K = K - np.dot(unoM, K) - np.dot(K, unoM) + np.dot(unoM, np.dot(K, unoM))
    # Autovalores y autovectores
    w, alpha = descending_eig(K)
    lambdas = w
    for col in range(alpha.shape[1]):
        alpha[:, col] = alpha[:, col] / np.sqrt(lambdas[col])

    # pre-proyección
    improypre = np.dot(K.T, alpha)
    proy_training = improypre[:, 0:args.eigenfaces]

else:

    # CARA MEDIA
    meanimage = np.mean(images_training, 0)
    images_training = [images_training[k, :] - meanimage for k in range(images_training.shape[0])]
    # PCA
    images_training = np.asarray(images_training)
    eigen_values, V = my_svd(images_training)

    B = V[0:args.eigenfaces, :]
    proy_training = np.dot(images_training, B.T)

clf.fit(proy_training, person_training.ravel())


# TEST PICTURES
test_path = args.face_test_directory
while(True):
    print("Input face path")
    picture_path = stdin.readline().rstrip().split()[0]
    if args.kernel:
        a = np.reshape((im.imread(test_path + picture_path + '.pgm') - 127.5) / 127.5, [1, areasize])
        unoML = np.ones([1, trnno]) / trnno
        Ktest = (np.dot(a, images_training.T) / trnno + 1) ** degree
        Ktest = Ktest - np.dot(unoML, K) - np.dot(Ktest, unoM) + np.dot(unoML, np.dot(K, unoM))
        imtstproypre = np.dot(Ktest, alpha)
        proy_test = imtstproypre[:, 0:args.eigenfaces]
    else:
        a = np.reshape(imageio.imread(test_path + picture_path + '.pgm') / 255.0, [1, areasize])
        a -= meanimage
        proy_test = np.dot(a, B.T)
    prediction = clf.predict(proy_test)
    print(names_dictionary[prediction[0]//1])