import os 
from scipy import ndimage as im
import numpy as np


def getImages(path, horizontal, vertical, figuresPerPerson, peopleNo , trainingNo, testNo):

	directories = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

	area = horizontal * vertical

	train = np.zeros([peopleNo * trainingNo, area])
	test = np.zeros([peopleNo * trainingNo, area])

	trainNames = {}
	testNames = {}

	training_img = 0 
	test_img = 0
	person_img = 0 
	dir_no = 0

	for d in directories: 
		for k in range(1, figuresPerPerson + 1):
			a = im.imread(path + '/' + d + '/{}'.format(k) + '.pgm')
			if person_img < trainingNo:
				train[training_img, :] = (np.reshape(a, [1, area])-127.5)/127.5
				trainNames.append(str(d))
				training_img+=1
			# this could be done out of this function
			else: 
				test[test_img, :] = (no.reshape(a, [1, area]) - 127.5)/127.5
				testNames.append(str(d))
				test_img+=1
			person_img+=1 
		dir_no += 1	
		if dir_no > peopleNo - 1:
			break
		person_img = 0 

	return train, test, trainNames, testNames