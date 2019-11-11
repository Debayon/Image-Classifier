import os
from PIL import Image
# import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd
import torch
import torch.utils.data as utils
import torch.nn as nn

## Get the list of all images 
files = os.listdir("dataset/val/cats")
files.remove(".DS_Store")
print(files[0])

## list of images
Images_train = []
Images_val = []

############### IF IT'S A CAT, it's '0' and IF IT'S A DOG, it's '1' #######################

# Labels = []



## defining required transforms or preprocessing on the images
data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

## reading the images applying the transformations, converting each of them to pytorch tensors and storing them in images list
for i in files:
	image = os.path.join("dataset/val/cats",i)
	im = Image.open(image)
	# imm = np.asarray(im)
	im = data_transforms(im)
	# Images.append(im)
	# Labels.append(0)
	Images_val.append([im, 0])


files = os.listdir("dataset/train/cats")
files.remove(".DS_Store")
print(files[0])


for i in files:
	image = os.path.join("dataset/train/cats",i)
	im = Image.open(image)
	# imm = np.asarray(im)
	im = data_transforms(im)
	# Images.append(im)
	# Labels.append('cat')
	Images_train.append([im, 0])


files = os.listdir("dataset/val/dogs")
files.remove(".DS_Store")
print(files[0])


for i in files:
	image = os.path.join("dataset/val/dogs",i)
	im = Image.open(image)
	# imm = np.asarray(im)
	im = data_transforms(im)
	# Images.append(im)
	# Labels.append('dog')
	Images_val.append([im, 1])


files = os.listdir("dataset/train/dogs")
files.remove(".DS_Store")
print(files[0])



for i in files:
	image = os.path.join("dataset/train/dogs",i)
	im = Image.open(image)
	# imm = np.asarray(im)
	im = data_transforms(im)
	# Images.append(im)
	# Labels.append('dog')
	Images_train.append([im, 1])




# df = pd.DataFrame(Images, columns=['Image', 'Label'])
# print(df.head())

print(Images_val[0])
print(Images_val[0][0])
print(Images_val[0][0][0])

print(len(Images_train))


batch_size = 20
# n_iters = 800
# num_epochs = n_iters / (len(Images_train) / batch_size)
# num_epochs = 10
#tensor_x = torch.stack([torch.Tensor(i) for i in Images_val])
train_loader = torch.utils.data.DataLoader(dataset=Images_train, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=Images_val, 
                                          batch_size=batch_size, 
                                          shuffle=False)


class SimpleCNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.conv2 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        
        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 2)
        
    
    def forward(self, x):        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        
        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * 16 *16)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return(x)


model = SimpleCNN()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iter = 0
for epoch in range(0, 13):
    print("epoch: ",epoch)
    for i, (images, labels) in enumerate(train_loader):
        # Load images
        # images = images.requires_grad_()
        images = images.requires_grad_().to(device)
        labels = labels.to(device)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1
        # print(iter)

        if iter % 200 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Load images
                # images = images.requires_grad_()
                images = images.requires_grad_().to(device)
                labels = labels.to(device)

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                # correct += (predicted == labels).sum()
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()


            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
            # print("hello")
