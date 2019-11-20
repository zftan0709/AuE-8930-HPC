from models import ResNet18
import os
import torch
from torch import optim, nn
from torchvision import transforms, datasets
import torchvision
import common
import matplotlib.pyplot as plt 

#DIRECTORY SETTINGS
os.chdir(".")#Go up two directories
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'base.pt')

#HYPERPARAMETERS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS=200
BATCH_SIZE = 64
criterion = nn.CrossEntropyLoss()
ADAM_OPTIMISER=True
LEARNING_RATE=0.0001
TESTING=True

train_transforms = transforms.Compose([# Data Transforms
                           #transforms.Resize(256),#Resize
                           transforms.RandomHorizontalFlip(),#Flip
                           transforms.RandomRotation(10),#Roatate
                           #transforms.RandomCrop(256),
                           transforms.RandomCrop(32, padding=4),
                           transforms.ToTensor(),#Convert to Tensor
                           transforms.Normalize([0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])#Normalize
                           #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                       ])

test_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                       ])



train_data = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=train_transforms)#Use CIFAR10 to train
train_data, valid_data = torch.utils.data.random_split(train_data, [int(len(train_data)*0.9), len(train_data) - int(len(train_data)*0.9)])
test_data = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=test_transforms)

print("Number of training examples: %i" % (len(train_data)))
print("Number of validation examples: %i" % (len(valid_data)))
print("Number of testing examples: %i" % (len(test_data)))

# Iterator
train_iterator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
test_iterator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

print("Loading models...")
# Model
#model = torchvision.models.resnet18(pretrained=True)#TorchVision
#for param in model.parameters():
#    param.requires_grad = False
#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, 10)
model = ResNet18()

# Enable Parallel Data Processing if more than two gpu exist
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    
model = model.to(device)

#Hyperparameters
if(ADAM_OPTIMISER):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
else:
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.5,weight_decay=5e-4)

train_loss_list = []
valid_loss_list = []
train_acc_list = []
valid_acc_list = []

EPOCHS_list = list(range(1,EPOCHS+1))
print("Start training...")
#Train
best_valid_loss = float('inf')
for epoch in range(EPOCHS):#Range of Epochs
    train_loss, train_acc = common.train(model, device, train_iterator, optimizer, criterion)#Train Loss Calculation
    valid_loss, valid_acc = common.evaluate(model, device, valid_iterator, criterion)#Validation Loss Calculation

    if valid_loss < best_valid_loss:#Validation Loss - Is current lower than the saved validation loss.
        best_valid_loss = valid_loss#Save the best loss (lowest)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)#Save the model
    print("| Epoch: %02i | Train Loss: %.3f | Train Acc: %05.2f | Val. Loss: %.3f | Val. Acc: %05.2f |" % (epoch+1,train_loss,train_acc*100,valid_loss,valid_acc*100))
    
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)
    
print("Finished training...")


#3. OUTPUT
if TESTING:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH)) #Load best weights from file
    test_loss, test_acc = common.evaluate(model, device, valid_iterator, criterion) #Test Loss is dependent on
    print("| Test Loss: %.3f | Test Acc: %05.2f" % (test_loss,test_acc*100))
    
#4. Plot    
plt.switch_backend('agg')
plt.plot(EPOCHS_list,train_loss_list,EPOCHS_list,valid_loss_list)
plt.legend(['Train Loss','Valid Loss'])
plt.grid(axis='y')
plt.xlabel('Number of Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.show()
plt.savefig('./models/loss.png')

plt.clf()
plt.plot(EPOCHS_list,train_acc_list,EPOCHS_list,valid_acc_list)
plt.legend(['Train Accuracy','Valid Accuracy'])
plt.grid(axis='y')
plt.xlabel('Number of Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()
plt.savefig('./models/acc.png')