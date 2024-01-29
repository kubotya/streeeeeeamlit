import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

classes_ja = ["Tシャツ/トップ", "ズボン", "プルオーバー", "ドレス", "コート", "サンダル", "ワイシャツ", "スニーカー", "バッグ", "アンクルブーツ"]
classes_en = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
n_class = len(classes_ja)
img_size = 28

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64*4*4, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.bn2(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, 64*4*4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
net = Net()
net.load_state_dict(torch.load("model_cnn.pth", map_location=torch.device("cpu")))

def predict(img):
    img = img.convert("L")
    img = img.resize((img_size, img_size))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0), (1.0))])
    img = transform(img)
    x = img.reshape(1, 1, img_size, img_size)

    net.eval()
    y = net(x)

    y_prob = torch.nn.functional.softmax(torch.squeeze(y))
    sorted_prob, sorted_inddices = torch.sort(y_prob, descending=True)
    return [(classes_ja[idx], classes_en[idx], prob.item()) \
            for idx, prob in zip(sorted_inddices, sorted_prob)]
