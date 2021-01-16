"""
 This file showcases some preporcesing techniques on our data with the aid of opencv2 mainly finding contour,Erosion,Dilation resizing
 images as it would be a perfect operation before giving it to our model
"""
import dataloader
import cv2
import numpy as np
"""
 preprocesing data
"""
def preprocess():

    pass

if __name__ == "__main__":
    path = "/media/anand/polyglot/BraindataPY"
    label_dict = {"no":1,"yes":0}
    train_dirlist = dataloader.train_or_testset("train/")
    test_dirlist = dataloader.train_or_testset("test/")
    #extracting path,images and labels from training data
    label_train = []      ###declare a list for tracking target labels for training datas
    images_train = []      ####assigning a list and storing image arrays of training images
    image_paths_train = [] ###This is an optional step to keep a track on the train image datas path.
    dataloader.create_data_loader(train_dirlist,label_train,images_train,image_paths_train)        # dirlist[:-1] includes no and yes directories
    print("train_set some samples")
    print(label_train[2])
    print(images_train[2])
    print(image_paths_train[2])
    for i in images_train:
        print(i)
    #extracting path,images and labels from testing data
    label_test = []      ###declare a list for tracking target labels for testing datas
    images_test = []      ####assigning a list and storing image arrays of testing images
    image_paths_test = [] ###This is an optional step to keep a track on the test image datas path
    dataloader.create_data_loader(test_dirlist,label_test,images_test,image_paths_test)
    print("test_set some samples")
    print(label_test[2])
    print(images_test[2])
    print(image_paths_test[2])

    """Creating shuffle trained dataset"""
    label_train,images_train,image_paths_train = dataloader.shuffle_data(label_train,images_train,image_paths_train)
    print(label_train[0:8])
    print(image_paths_train[0:8])
    """Creating shuffle test dataset"""
    label_test,images_test,image_paths_test = dataloader.shuffle_data(label_test,images_test,image_paths_test)
    print(label_test[1:12])
    print(image_paths_test[2:7])
