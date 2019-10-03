import os
import imageio
from os import listdir
from os.path import join, isdir
from scipy import ndimage as im
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np

def getImages(path, horizontal, vertical, figuresPerPerson, peopleNo , trainingNo, testNo):

	directories = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

	area = horizontal * vertical

	train = np.zeros([peopleNo * trainingNo, area])
	test = np.zeros([peopleNo * trainingNo, area])

	trainNames = []
	testNames = []

	training_img = 0 
	test_img = 0
	person_img = 0 
	dir_no = 0

	for d in directories: 
		for k in range(1, figuresPerPerson + 1):
			a = imageio.imread(path + '/' + d + '/{}'.format(k) + '.pgm')
			if person_img < trainingNo:
				train[training_img, :] = (np.reshape(a, [1, area])-127.5)/127.5
				trainNames.append(str(d))
				training_img+=1
			# this could be done out of this function
			else: 
				test[test_img, :] = (np.reshape(a, [1, area]) - 127.5)/127.5
				testNames.append(str(d))
				test_img+=1
			person_img+=1 
		dir_no += 1	
		if dir_no > peopleNo - 1:
			break
		person_img = 0 

	return train, test, trainNames, testNames

def openImages(path, personno, trnperper, areasize):
    onlydirs = [f for f in listdir(path) if isdir(join(path, f))]
    images_no = personno * trnperper
    images = np.zeros([images_no, areasize])
    person = np.zeros([images_no, 1])
    imno = 0
    per = 1
    names = {}
    for dire in onlydirs:
        for k in range(1, trnperper + 1):
            a = imageio.imread(path + dire + '/{}'.format(k) + '.pgm') / 255.0
            images[imno, :] = np.reshape(a, [1, areasize])
            person[imno, 0] = per
            imno += 1
        names[per] = dire
        per += 1
        if per > personno:
            break
    return images, person, names