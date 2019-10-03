import os
from os import listdir
from os.path import join, isdir

import imageio
import numpy as np


def getImages(path, horizontal, vertical, figuresPerPerson, peopleNo, trainingNo, testNo):
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
                train[training_img, :] = (np.reshape(a, [1, area]) - 127.5) / 127.5
                trainNames.append(str(d))
                training_img += 1
            # this could be done out of this function
            else:
                test[test_img, :] = (np.reshape(a, [1, area]) - 127.5) / 127.5
                testNames.append(str(d))
                test_img += 1
            person_img += 1
        dir_no += 1
        if dir_no > peopleNo - 1:
            break
        person_img = 0

    return train, test, trainNames, testNames


def openImages(path, peopleno, trainingsperperson, area):
    directories = [f for f in listdir(path) if isdir(join(path, f))]
    images_no = peopleno * trainingsperperson
    person = np.zeros([images_no, 1])
    images = np.zeros([images_no, area])
    names = {}
    per = 1
    imgno = 0
    for dire in directories:
        for k in range(1, trainingsperperson + 1):
            a = imageio.imread(path + dire + '/{}'.format(k) + '.pgm') / 255.0
            images[imgno, :] = np.reshape(a, [1, area])
            person[imgno, 0] = per
            imgno += 1
        names[per] = dire
        per += 1
        if per > peopleno:
            break
    return images, person, names
