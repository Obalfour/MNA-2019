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


trainFigures, testFigures, trainingNames, testNames = getImages(FIGURE_PATH, HORIZONTAL, VERTICAL, FIGURES_PER_PERSON, PEOPLE_NO, TRAINING_NO, TEST_NO)

print(trainFigures)
print(testFigures)
print(trainingNames)
print(testNames)