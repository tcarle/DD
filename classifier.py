import numpy as np
import torch
import torchvision
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from time import time
from torchvision import datasets, transforms
from torch import nn, optim



def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show(block=True)






transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)


##################################3
# Print images

# dataiter = iter(valloader)
# images, labels = dataiter.next()

for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    for j in img:
        print("I ")
        for l in j:
            print(str(l.item()), end = ' ')
        print("\n")


  print("R")
    
  for i in labels:
    print(str(i.item()), end = ' ')
  break



#dataiter = iter(trainloader)
#images, labels = dataiter.next()

#print(images.shape)
#print(labels.shape)

#plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

#model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
#                      nn.ReLU(),
#                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
#                      nn.ReLU(),
#                      nn.Linear(hidden_sizes[1], output_size),
#                      nn.ReLU())
                      #nn.LogSoftmax(dim=1))
#print(model)

criterion = nn.CrossEntropyLoss()#nn.NLLLoss()
images, labels = next(iter(trainloader))

images = images.view(images.shape[0], -1)




# logps = model(images) #log probabilities
# loss = criterion(logps, labels) #calculate the NLL loss

# optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
# time0 = time()
# epochs = 15
# for e in range(epochs):
#     running_loss = 0
#     for images, labels in trainloader:
#         # Flatten MNIST images into a 784 long vector
#         images = images.view(images.shape[0], -1)
    
#         # Training pass
#         optimizer.zero_grad()
        
#         output = model(images)
#         loss = criterion(output, labels)
        
#         #This is where the model learns by backpropagating
#         loss.backward()
        
#         #And optimizes its weights here
#         optimizer.step()
        
#         running_loss += loss.item()
#     else:
#         print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
# print("\nTraining Time (in minutes) =",(time()-time0)/60)

        
# images, labels = next(iter(valloader))

# img = images[0].view(1, 784)

# with torch.no_grad():
#     logps = model(img)

# ps = torch.exp(logps)
# probab = list(ps.numpy()[0])
# print("Predicted Digit =", probab.index(max(probab)))


# torch.save(model.state_dict(), './my_mnist_model.pt') 

########################################################################################

# Load model and print

model2 = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.ReLU())

model2.load_state_dict(torch.load('./my_mnist_model.pt'))

#view_classify(img.view(1, 28, 28), ps)
correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model2(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    print(probab)
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1
  break
print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))


# i=0
# j=0
# k=0
# for param in model2.parameters():
#     k+=1

# print("S: " +str(int(k/2)))

# for param in model2.parameters():
# #    print(param.data)
    
#     if param.ndim == 2:
#         print("L " + str(i) + ": " +str(len(param)) + " " +str(len(param[0])))
#     else:
#         print("B " + str(i) + ": " +str(len(param)))
#     for i_inter in range(len(param.data)):
#         inter = param.data[i_inter]
#         if param.ndim == 2:
#             for i_inter2 in range(len(inter)):
#                 print(str(inter[i_inter2].item()), end='')
#                 if i_inter2 < len(inter)-1:
#                     print(" ", end='')
#         else:
#             print(str(inter.item()), end='')
#         if i_inter < len(param.data)-1:
#             print(" ", end='')
#     print("\n")
#     if j == 1:
#         i+=1
#         j=0
#     else:
#         j+=1
