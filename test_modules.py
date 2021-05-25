import unittest
import pytest
from dataloader import create_data_loader,shuffle_data,train_or_testset
from preprocessing import fin_preprocess,preprocess
import torch
import model 
from model import CNN
import warnings
import numpy as np
warnings.filterwarnings("ignore")

path = "/media/anand/polyglot/BraindataPY"
label_dict = {"no":1,"yes":0}
train_dirlist = train_or_testset("train/")
test_dirlist = train_or_testset("test/")
#extracting path,images and labels from training data
label_train = []      ###declare a list for tracking target labels for training datas
images_train = []      ####assigning a list and storing image arrays of training images
image_paths_train = [] ###This is an optional step to keep a track on the train image datas path.
label_test = []      ###declare a list for tracking target labels for testing datas
images_test = []      ####assigning a list and storing image arrays of testing images
image_paths_test = [] ###This is an optional step to keep a track on the test image datas path
model = CNN()

X = [] #Final image feature map


"""================= Testing Modules ================="""



class TestBraindata(unittest.TestCase):
    
    def test_create_dataloader4train(self):
        create_data_loader(train_dirlist,label_train,images_train,image_paths_train)
        assert len(label_train) == 196
    def test_create_datafolder4test(self):
        create_data_loader(test_dirlist,label_test,images_test,image_paths_test)
        assert len(image_paths_test)== 57

    def test_shuffle_func(self):
        shuffle_data(label_train,images_train,image_paths_train)
        assert len(label_train)== 196
        shuffle_data(label_test,images_test,image_paths_test)
        assert len(label_test)== 57

    
    def test_fin_process(self):
        X_train,y_train = fin_preprocess(image_paths_train,X,label_train)

        assert X_train.shape == (196, 224, 224, 3)
    def test_pytorch_model(self):
        x = torch.randn(64,3,224,224)
        assert model(x).shape == torch.Size([64,2])

        

if __name__ == "__main__":
    unittest.main()
