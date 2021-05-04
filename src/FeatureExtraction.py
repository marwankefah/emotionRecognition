import Ada as Ada
import Helper as hf
import numpy as np
import torch
from torchvision import transforms

#Path To Dataset
path = ""
#Path to Cifar
cifarpath = ""

train_transforms = transforms.Compose([transforms.Resize(22),
                                       transforms.CenterCrop(22),
                                       transforms.Grayscale(),
                                       transforms.ToTensor()])

trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(22),
                            transforms.CenterCrop(22),
                            transforms.Grayscale(),
                            transforms.ToTensor()])

trainloader = hf.loadData(path, train_transforms, batchsize=5000)
cifarloader = hf.loadcifar(cifarpath, train_transforms, batchsize=7000)



dataiter = iter(trainloader)
images, labels = dataiter.next()

cifariter = iter(cifarloader)
images2, labels2 = cifariter.next()

real, imag = hf.build_filters()

X1 = torch.cat((images[0:2500], images2[0:3500])).numpy().reshape(-1, 22, 22)
Y1 = np.ones((6000, 1))
Y1[2500::] = 0


X2 = torch.cat((images[2500::], images2[3500::])).numpy().reshape(-1, 22, 22)
Y2 = np.ones((6000, 1))
Y2[2500::] = 0


ada = Ada.AdaBoostSelection(200)
X_feat1 = hf.GaborFeatures(X1, real, imag)
X_feat2 = hf.GaborFeatures(X2, real, imag)
ada.fit(X_feat1, Y1)

acctrain = ada.validate(X_feat1, Y1)
accvalidate = ada.validate(X_feat2, Y2)


print("Training Accuracy = {0} ".format(acctrain))
print("Validation Accuracy = {0} ".format(accvalidate))


a = []
for classifier in ada.Classifiers:
    a.append(classifier.feature_index)

col = {"x": a}
df = pd.DataFrame(col)
df.insert(0, "x", a, True)
df.to_csv()

print(1)
