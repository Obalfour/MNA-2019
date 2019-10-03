from os import listdir
from os.path import join, isdir
import numpy as np
from sklearn import svm
import imageio
from sys import stdin
from utils import *
from matrixtools import *
import configparser

config = configparser.ConfigParser()
config.read("./configFile.ini")

# [Size]
HORIZONTAL= config.getint("Size", "HORIZONTAL") #92
VERTICAL= config.getint("Size", "VERTICAL") #112

# [Image]
IMAGES_PER_PERSON = config.getint("Image", "IMAGES_PER_PERSON") #10
PEOPLE_NO = config.getint("Image", "PEOPLE_NO") #40
IMAGE_PATH = config.get("Image", "IMAGE_PATH") #../att_faces/

# [Training]
TEST_NO = config.getint("Training", "TEST_NO") #4
TRAINING_NO = config.getint("Training", "TRAINING_NO") #6

# [Method]
METHOD = config.get("Method", "METHOD")

# [Test]
QUERY = config.get("Test","QUERY")
EIGEN_FACES = config.getint("Test", "EIGEN_FACES") #60

clf = svm.LinearSVC()
# TRAINING

print("Starting image recognition, wait until you are asked to write the image path")
images_training, person_training, names = openImages(IMAGE_PATH, PEOPLE_NO, TRAINING_NO, HORIZONTAL*VERTICAL)
if METHOD == 'KPCA':
    images_training *= 255.0
    images_training -= 127.5
    images_training /= 127.5

    # KERNEL: polinomial de grado degree
    degree = 2
    K = (np.dot(images_training, images_training.T) / (PEOPLE_NO * TRAINING_NO) + 1) ** degree
    # esta transformación es equivalente a centrar las imágenes originales...
    unoM = np.ones([(PEOPLE_NO * TRAINING_NO), (PEOPLE_NO * TRAINING_NO)]) / (PEOPLE_NO * TRAINING_NO)
    K = K - np.dot(unoM, K) - np.dot(K, unoM) + np.dot(unoM, np.dot(K, unoM))
    # Autovalores y autovectores
    w, alpha = descending_eig(K)
    lambdas = w
    for col in range(alpha.shape[1]):
        alpha[:, col] = alpha[:, col] / np.sqrt(lambdas[col])

    # pre-proyección
    improypre = np.dot(K.T, alpha)
    proy_training = improypre[:, 0:EIGEN_FACES]

else:

    # CARA MEDIA
    meanimage = np.mean(images_training, 0)
    images_training = [images_training[k, :] - meanimage for k in range(images_training.shape[0])]
    # PCA
    images_training = np.asarray(images_training)
    eigen_values, V = my_svd(images_training)

    B = V[0:EIGEN_FACES, :]
    proy_training = np.dot(images_training, B.T)

clf.fit(proy_training, person_training.ravel())


# TEST PICTURES
test_path = QUERY
while(True):
    print("Input face path")
    image_path = stdin.readline().rstrip().split()[0]
    if METHOD == 'KPCA':
        a = np.reshape((imageio.imread(test_path + image_path + '.pgm') - 127.5) / 127.5, [1, VERTICAL*HORIZONTAL])
        unoML = np.ones([1, (PEOPLE_NO * TRAINING_NO)]) / (PEOPLE_NO * TRAINING_NO)
        Ktest = (np.dot(a, images_training.T) / (PEOPLE_NO * TRAINING_NO) + 1) ** degree
        Ktest = Ktest - np.dot(unoML, K) - np.dot(Ktest, unoM) + np.dot(unoML, np.dot(K, unoM))
        imtstproypre = np.dot(Ktest, alpha)
        proy_test = imtstproypre[:, 0:EIGEN_FACES]
    else:
        a = np.reshape(imageio.imread(test_path + image_path + '.pgm') / 255.0, [1, VERTICAL*HORIZONTAL])
        a -= meanimage
        proy_test = np.dot(a, B.T)
    prediction = clf.predict(proy_test)
    print(names[prediction[0]//1])