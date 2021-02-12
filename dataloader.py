"""                                                                                                 
  This module is to load all those 'no' and 'yes' image files of brain tumour , as per the regular   
  convenient manner of image classification use case. We will be making our labels according to the  
  yes/no filenames from the directory.                                                               
"""        
import cv2
import numpy as np
import os ,sys
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings("ignore")

"""                                                                                                 
 Image file paths for this system.                                                                  
""" 
path = "/media/anand/polyglot/BraindataPY/"

"""creating a data loading function """

def train_or_testset(x):
    yes = os.path.join(path,x,"yes/")
    no = os.path.join(path,x,"no/")
    dirlist = [no,yes]
    return dirlist

""" checking on actual file system."""

def show_file_details(directories):
    for dir in directories:
        print(len(os.listdir(dir)),"files in {} directory".format(dir.split("/")[-2]))

"""
displaying some sample images
"""
def display_image(directory,index):
    for file_name in os.listdir(directory[0])[:index]:
        image = cv2.imread(directory[0]+'/'+file_name)
        #image = cv2.imread(os.path.join(dirlist[0],file_name))
        plt.imshow(image)
        plt.show()


label_dict = {"no":0, "yes":1}
def create_data_loader(dirlis,labellist,image_dtlst,image_pathlst):
    for files in dirlis:
        for j in os.listdir(files):
            image_path = os.path.join(files,j)
            labels = files.split("/")[-2]    
            #print(labels)
            labellist.append(label_dict[labels])          ## appending labels
            file_path = os.path.join(files,j)
            image_pathlst.append(file_path)      ##Printing it
            #print(os.path.join(files,j))
            image = cv2.imread(image_path)
            image_array = np.array(image)   ## Its already a numpy array so we choose not to append this
            image_dtlst.append(image)
    print(len(labellist),":-sample label to show where actually it comes from",labels)
    print(len(image_pathlst),":-sample path to show where actually it comes from",image_path)
    print(len(image_dtlst))

"""
Before giving the preprocessed data  to our ML-Model we need to reshuffle our series patterned data ,
otherwise model will learn a specific pattern from the dataset and carry on the similar behavior to the test dataset also.
"""
def shuffle_data(labelset,img_dataset,pathset):                                                     
    shuf_idx = [i for i in range(0,len(labelset))]    ##lst = [i for i in range(0,len(trainlbl))]                                                               
    labelset = np.array(labelset)                     ## random.shuffle(lst)
    img_dataset = np.array(img_dataset,dtype="object")               ## sett = trainlbl[lst]                                        
    pathset = np.array(pathset)                       ##  settpath = trainpath[lst]                                                                                     
    random.shuffle(shuf_idx)                          ##
    labelset = labelset[shuf_idx]
    img_dataset = img_dataset[shuf_idx]                                                         
    pathset = pathset[shuf_idx] 
    return labelset,img_dataset,pathset


if __name__ == "__main__":
    train_dirlist = train_or_testset("train/")
    test_dirlist = train_or_testset('test/')
    print("for train set")
    print(show_file_details(train_dirlist)) 
    print("for test set")
    print(show_file_details(test_dirlist))
    print(display_image(train_dirlist,1))
    print(display_image(test_dirlist,1))
    #extracting path,images and labels from training data
    label_train = []      ###declare a list for tracking target labels for training datas
    images_train = []      ####assigning a list and storing image arrays of training images
    image_paths_train = [] ###This is an optional step to keep a track on the train image datas path.
    create_data_loader(train_dirlist,label_train,images_train,image_paths_train)        # dirlist[:-1] includes no and yes directories
    print("train_set some samples")
    print(label_train[2])
    print(images_train[2])
    print(image_paths_train[2])
    #extracting path,images and labels from testing data
    label_test = []      ###declare a list for tracking target labels for testing datas
    images_test = []      ####assigning a list and storing image arrays of testing images
    image_paths_test = [] ###This is an optional step to keep a track on the test image datas path
    create_data_loader(test_dirlist,label_test,images_test,image_paths_test)
    print("test_set some samples")
    print(label_test[2])
    print(images_test[2])
    print(image_paths_test[2])
    
    """Creating shuffle trained dataset"""
    label_train,images_train,image_paths_train = shuffle_data(label_train,images_train,image_paths_train)
    print(label_train[0:8])
    print(image_paths_train[0:8])
    """Creating shuffle test dataset"""
    label_test,images_test,image_paths_test = shuffle_data(label_test,images_test,image_paths_test)
    print(label_test[1:12])
    print(image_paths_test[2:7])

