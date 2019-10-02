# for python3 use configparser, it has been changed from one version to another
import configparser
from utils import *

config = configparser.ConfigParser()
config.read("./configFile.ini")

# [Size]
HORIZONTAL = config.getint("Size", "HORIZONTAL")
VERTICAL = config.getint("Size", "VERTICAL")

# [Figure]
FIGURES_PER_PERSON = config.getint("Figure", "FIGURES_PER_PERSON")
PEOPLE_NO = config.getint("Figure", "PEOPLE_NO")
FIGURE_PATH = config.get("Figure", "FIGURE_PATH")	# Path to the directory with the faces

# [Training]
TEST_NO = config.getint("Training", "TEST_NO")
TRAINING_NO = config.getint("Training", "TRAINING_NO") #How many photos used for training out of 10 [1,2,3,4,5,6,7,8,9,10]

# [Method]
METHOD = config.get("Method", "METHOD")

# [Test]
QUERY = config.get("Test", "QUERY")
EIGEN_FACES = config.getint("Test", "EIGEN_FACES")

training = TRAINING_NO * PEOPLE_NO
test = TEST_NO * PEOPLE_NO

trainFigures, testFigures, trainingNames, testNames = getImages(FIGURE_PATH, HORIZONTAL, VERTICAL, FIGURES_PER_PERSON, PEOPLE_NO, TRAINING_NO, TEST_NO)

# KERNEL 
if METHOD == 'KPCA':

	#Polinomial of 2 degree
	degree = 2
    A = (np.dot(tralinFigures, trainFigures.T) / training + 1) ** degree

    # esta transformación es equivalente a centrar las imágenes originales... 
    # hacemos el test y el entrenamiento a la vez, para obtener el resultado final
    unoM = np.ones([training, training]) / training
    unoMTest = np.ones([test, training]) / training
    A = A - np.dot(unoM, A) - np.dot(A, unoM) + np.dot(unoM, np.dot(A, unoM))

    Atest = (np.dot(testFigures, trainFigures.T) / training + 1) ** degree
    Atest = Atest - np.dot(unoMTest, A) - np.dot(Atest, unoMTest) + np.dot(unoMTest, np.dot(A, unoM))

    # Autovalores y autovectores
    w, alpha = my_eig(A)
    lambdas = w

    for col in range(alpha.shape[1]):
        alpha[:, col] = alpha[:, col] / np.sqrt(lambdas[col])

    # pre-proyección
    improypre = np.dot(A.T, alpha)
    improypreTest = np.dot(Atest, alpha)
    
    nmax = alpha.shape[1]
    accs = np.zeros([nmax, 1])



