import cv2
import glob
import torch 
import numpy as np
import matplotlib.pyplot as plt
from torch import nn                        
from PIL import Image 
from importlib.resources import path
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor 
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot


g_img_path = '/home/park/coding/study/FMNIST_Tutorial/custom dataset/bag.png'

class MyModel(nn.Module):                        
    def __init__(self):                           
        super(MyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(                                                   
            nn.Linear(28*28, 512),                
            nn.ReLU(),                           
            nn.Linear(512, 512),                 
            nn.ReLU(),                            
            nn.Linear(512, 10)                   
        )
    def forward(self, x):                     
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



class CustomDataSet(Dataset):
    # 데이터셋 전처리
    global g_img_path
    
    def __init__(self, transform =None):
        self.trans = transforms.Compose([transforms.Resize((28,28)),
                                    transforms.ToTensor()])
        
        self.test_data_list = glob.glob(g_img_path)
        self.transform = transform
        
        self.test_class_list = [8] * len(self.test_data_list) #라벨링 해주는 부분
        '''
        0"T-shirt/top",
        1"Trouser",
        2"Pullover",
        3"Dress",
        4"Coat",
        5"Sandal",
        6"Shirt",
        7"Sneaker",
        8"Bag",
        9"Ankle boot",
        '''
        
        
        #print("test_data_list : " , self.test_data_list)
        #print("test_class_list : " , self.test_class_list)
                
    # 데이터셋 길이, 샘플의 수 지정
    def __len__(self):
        return len(self.test_data_list)
    
    #데이터셋에서 특정 1개의 샘플을 가져오는 함수
    def __getitem__(self, idx):
        img_path = self.test_data_list[idx]
        label = self.test_class_list[idx]
        img = Image.open(img_path)
        
        if self.transform is not None:
            img = img.resize((28,28))
            img = self.transform(img)
                
        #print("img : ", img)
        #print("label : ", label)
        return img, label

batch_size = 1

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataSet(transform=transform)    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    #for Image, Label in dataloader:                                             # DataLoader 순회 후 Train_feature(이미지)와 Train_label(라벨)의 묶음(Batch)을 반환한다.
    #    print(f"Shape of Image [N, C, H, W]: {Image.shape} {Image.dtype}")           # Image 반환 내용 N(Batch), C(Channel), H(Height), W(width)
    #    print(f"Shape of Label: {Label.shape} {Label.dtype}")                        # Label 반환 내용 
    #    break
    #print("dataset : ", dataset)
    #print("dataloader : ", dataloader)
    #print("len : ", len(dataset))
 
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = MyModel()
model_path = '/home/park/coding/study/FMNIST_Tutorial/mymodel.pth'
model.load_state_dict(torch.load(model_path))


'''
for epoch in range (1):
    print(epoch)
    for batch_size in dataloader:
        img, label = batch_size
        print("imgsize and label: ", img.size(), label)
'''    
        
model.eval()
x , y = dataset[0][0], dataset[0][1]
#x = transforms
#print("x의 데이터 타입은 : {}, shape은 : {}, 데이터는 {} 입니다." .format(x.dtype, x.shape, x))
#print("x shape : ", x.shape)
#print("y shape : ", y)
with torch.no_grad():
    pred = model(x)
    predicted , actual = classes[pred[0].argmax(0)], classes[y] 
    #print(f'Predicted: "{predicted}", Actual: "{actual}"')     # 추론 결과 출력 
    print(f"입력하신 파일의 객체는 {predicted} 입니다.")

Original_img = cv2.imread(g_img_path)
Original_img = cv2.cvtColor(Original_img, cv2.COLOR_BGR2RGB)
Predicted_img = Image.open(g_img_path)
Predicted_img = Predicted_img.resize((28,28))
plt.subplot(121), plt.axis('on'), plt.imshow(Original_img), plt.title("Original Image")
plt.subplot(122), plt.axis('on'), plt.imshow(Predicted_img), plt.title("Classification : " + predicted)
plt.show()
