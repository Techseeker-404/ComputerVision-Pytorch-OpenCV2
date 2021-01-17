"""
 This file showcases some preporcesing techniques on our data with the aid of opencv2 mainly finding contour,Erosion,Dilation resizing
 images as it would be a perfect operation before giving it to our model
"""
import dataloader
import cv2
import numpy as np
import imutils
"""
 preprocesing data
"""
def preprocess(image):
    
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    #Converting the image to grayscale and blurring it a bit.
    grayscale = cv2.GaussianBlur(grayscale,(5,5),0)
    threshold_img = cv2.threshold(grayscale,45,255,cv2.THRESH_BINARY)[1]
    threshold_img = cv2.erode(threshold_img,None,iterations=2)
    threshold_img = cv2.dilate(threshold_img,None,iterations=2)#Find the contours in the image and grab out the largest one.
    contour = cv2.findContours(threshold_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    C = max(contour, key = cv2.contourArea)
    ext_left = tuple(C[C[:,:,0].argmin()][0]) #finding the extreme points
    ext_right = tuple(C[C[:,:,0].argmax()][0])# Lateral sections.
    ext_top = tuple(C[C[:,:,1].argmin()][0])# Top and Bottom
    ext_bot = tuple(C[C[:,:,1].argmax()][0])
    new_image = image[ext_top[1]:ext_bot[1],ext_left[0]:ext_right[0]]
    return new_image
"""
 next resizing, final level preprocessing before giving it to a model.
"""
def fin_preprocess(images,train_feature_lst,labels):
    for img in images:
        image = cv2.imread(img)
        image = preprocess(image)
        image = cv2.resize(image,(224,224),interpolation=cv2.INTER_CUBIC)
        image = image/255 #Normalizing images
        train_feature_lst.append(image)
    return np.array(train_feature_lst),np.array(labels)

if __name__ == "__main__":
    path = "/media/anand/polyglot/BraindataPY"
    label_dict = {"no":1,"yes":0}
    train_dirlist = dataloader.train_or_testset("train/")
    test_dirlist = dataloader.train_or_testset("test/")
    #extracting path,images and labels from training data
    label_train = []      ###declare a list for tracking target labels for training datas
    images_train = []      ####assigning a list and storing image arrays of training images
    image_paths_train = [] ###This is an optional step to keep a track on the train image datas path.
    dataloader.create_data_loader(train_dirlist,label_train,images_train,image_paths_train) # dirlist[:-1] includes no and yes directories
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
    """ Extracting Final training set"""
    X = [] #Final image feature map
    X_train,y_train = fin_preprocess(image_paths_train,X,label_train)
    
    print(X_train[2])
    print(y_train[2])
