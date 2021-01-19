""" This file is the main file where entire preprocessing, training and model evaluation takes place"""
import dataloader
import preprocessing
import model
from model import CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

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
    X_train,y_train = preprocessing.fin_preprocess(image_paths_train,X,label_train)
    
    """Extracting final test set"""
    X_testset = []
    X_test,y_test = preprocessing.fin_preprocess(image_paths_test,X_testset,label_test)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    print(model.parameters)
    #parameters(user defined)
    in_channels = 3
    num_classes = 2
    learning_rate = 0.001
    BATCH_SIZE = 66
    EPOCHS = 60
    momentum = 0.9
    # Loss and Loss_function criterion
    loss_function = nn.CrossEntropyLoss()   #As we are using cross entropy loss we dont have to use Softmax at the end
    optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    # For that we need to convert our train and test set data labels to torch tensors
    X_train = torch.tensor(X_train)
    X_train = X_train.permute(0,3,2,1)  ## VERY ESSENTIAL AS TORCH EXPECTS INPUT TO BE IN THE CHANNEL FIRST MANNER AS LIKE
    y_train = torch.tensor(y_train)             ## LIBTORCH
    X_test = torch.tensor(X_test)
    X_test = X_test.permute(0,3,2,1)
    y_test = torch.tensor(y_test)
    #ACTUAL TRAINING
    final_losses = []
    def train(model_net):
        
        for epoch in range(EPOCHS):# here as we have already seperated Features and labels ,we have to
            for i in range(0,len(X_train),BATCH_SIZE):# Initiate a for loop which steps or iter train data by the steps of
                #print(i,i+BATCH_SIZE)             #our user Defined BATCH_SIZE ,else if data is coming as a single unit,
                                                    #Then we can use inbuilt torch "Dataloader" and enumerate data and labels using a single for loop
                X_train_batch = X_train[i : i+BATCH_SIZE].to(device=device) #converting data into "CUDA" or device standards
                y_train_batch = y_train[i : i+BATCH_SIZE].to(device=device)
                #Forward pass
                y_pred_outs = model(X_train_batch.float())  #should be converted to float otherwise it will throw a RUNTIME error of expecting DOUBLE
                loss = loss_function(y_pred_outs,y_train_batch.long())
                final_losses.append(loss)
                if epoch % 2 ==1:
                    print("Epoch number:{} and loss:{}".format(epoch,loss.item()))
                #backward propagation
                optimizer.zero_grad()   #zero setting all the accumulated gradients
                loss.backward()
                optimizer.step()
    train(model)
    #plotting the loss value as 60 epochs runs for 3 batches it will be eventually 180 
    plt.plot(range(180),final_losses)
    plt.xlabel('Loss')
    plt.ylabel("Epochs")
    plt.show()
    #checking accuracy
    def check_accuracy(data,label,model):
#     if data.X_train:
#         print("checking accuracy on training data...")
#     else :
#         print("checking accuracy on testing data...")
        
        num_correct = 0
        num_samples = 0
        model.eval()
        with torch.no_grad():
            for i,datas in enumerate(data):
                datas = datas.unsqueeze(1)
                datas = datas.float()
                datas = datas.permute(1,0,2,3)
                
                
                x = datas.to(device=device)
                model_out = model(x)
                predicted_y = torch.argmax(model_out)
                predicted_y = predicted_y.item()
                #print(predicted_y)
                real_y = label[i].to(device=device)
                #real_y = label[i]
                real_y = real_y.item()
               
                if predicted_y == real_y:
                    num_correct += 1
                num_samples += 1
                
                accuracy = (num_correct/num_samples) * 100
                
                
            #print("accuracy : {}".format(accuracy))
            print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
 
            
        model.train()


    
    check_accuracy(X_test,y_test,model)
    
