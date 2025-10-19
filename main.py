import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms,models
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PneumoniaDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in ['NORMAL','PNEUMONIA']:
            class_dir = os.path.join(root_dir,label)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir,img_name))
                self.labels.append(0 if label == 'Normal' else 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img_path = self.image_paths[item]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[item]

        if self.transform:
            image = self.transform(image)
        return image,label


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

train_dataset = PneumoniaDataset(root_dir='data/train',transform=transform)
test_dataset = PneumoniaDataset(root_dir='data/test',transform=transform)
val_dataset = PneumoniaDataset(root_dir='data/val',transform=transform)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)
val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features,2)#NORMAL,PNEUMONIA
model = model.to(device)

