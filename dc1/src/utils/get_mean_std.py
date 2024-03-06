import numpy
from .load_labels import load_labels
import os
from torchvision import transforms

TRAIN_DIR = "../data/train/train"
TEST_DIR = "../data/test/test"


def compute_mean_std(dataset):
    # compute the mean and std of dataset

    data_r = numpy.dstack([dataset[i][0][0:, :, :] for i in range(len(dataset))])
    data_g = numpy.dstack([dataset[i][0][1:, :, :] for i in range(len(dataset))])
    data_b = numpy.dstack([dataset[i][0][2:, :, :] for i in range(len(dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std