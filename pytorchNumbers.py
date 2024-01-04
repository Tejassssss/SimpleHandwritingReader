import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)

class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Make a 3 layer convolutional neural network
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(),
            # 64 final layer, 10 potential answers 0-9
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x): 
        return self.model(x)

clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

def learn():
    for epoch in range(10):
        for batch in dataset: 
            x,y = batch 
            x, y = x.to('cuda'), y.to('cuda') 
            yhat = clf(x) 
            loss = loss_fn(yhat, y) 

            opt.zero_grad()
            loss.backward() 
            opt.step() 

        print(f"Epoch:{epoch} loss is {loss.item()}")


if __name__ == "__main__":
    learn()
    with open('model_state.pt', 'wb') as f: 
        save(clf.state_dict(), f)

    with open('model_state.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  

    tempImg = Image.open('img_1.jpg') 
    img_tensor = ToTensor()(tempImg).unsqueeze(0).to('cuda')
    tempImg = Image.open('img_2.jpg') 
    img_tensor2 = ToTensor()(tempImg).unsqueeze(0).to('cuda')
    tempImg = Image.open('img_3.jpg') 
    img_tensor3 = ToTensor()(tempImg).unsqueeze(0).to('cuda')

    print(torch.argmax(clf(img_tensor)))
    print(torch.argmax(clf(img_tensor2)))
    print(torch.argmax(clf(img_tensor3)))