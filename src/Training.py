import Classifier as cl
import Helper as hf
import torch


train_transforms = hf.defineTransforms()

#Path To Dataset
path1 = ""


# CSV file used to obtain the most important gabor features
path2 = '../model/a.csv'
#Path To model
modelpath = "../model/checkpoint.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model, optimizer, criterion= cl.load_checkpoint(modelpath, dropout=0.2, lr=0.001, device=device)
trainloader, testloader = hf.getProcessedData(path1, path2, 512, 512, transform=train_transforms)


valid_loss = cl.training(model, trainloader=trainloader, validloader=testloader, criterion=criterion,
                         optimizer=optimizer,modelPath=modelpath,device=device, epochs=300)

train, trainacc = cl.validation(model, trainloader, criterion, device)

testloss, testacc = cl.validation(model, testloader, criterion, device)

print("Training Acc :",trainacc)
print("Test Acc ",testacc)
