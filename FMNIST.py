
import numpy as np
import torch 
from torch import nn                        # 모델을 만들때 사용되는 base class 이며 클래스로 모델을 만들때는 nn, 함수로 바로 사용할때는 nn.functional을 사용한다. (편의에 따라 사용)
                                            # nn의 경우 weight를 사용하지만 nn.functional의 경우 filter를 사용하는 경우가 많다.(functional도 weight를 사용할 수 는 있다.) 
from torch.utils.data import DataLoader     # 데이터작업을 위한 기본요소 : dataloader - datasets를 순회 가능한 iterable 로 감싼다. datasets - 샘플과 정답을 저장한다. 
from torchvision import datasets            # torchvision의 datasets은 CIFAR, COCO등과 같은 비전 데이터를 갖고있다. (https://pytorch.org/vision/stable/datasets.html - 전체 데이터 셋)
from torchvision.transforms import ToTensor # TorchText, TorchVision, TorchAudio은 같이 도메인 특화 라이브러리로 데이터 셋을 제공한다. 
from torchviz import make_dot


# TorchVision의 Datasets 중 FasionMNIST 데이터 셋을 사용한다. 
training_data = datasets.FashionMNIST(  
    root="FashionMNIST",    # 학습/테스트 데이터가 저장 되는 위치
    train=True,             # 학습용, 테스트용을 결정한다. 
    download=True,          # 저장 위치에 데이터가 없는 경우 어떻게 할 것인가 선택 (True: 없는 경우 다운, False: 없어도 다운받지 않음)
    transform=ToTensor(),   # 특징(feature)와 정답(label) 변형(transform)을 지정한다.
                            # 샘플과 정답을 각각 변경하기 위한 파라미터 (모든 TorchVision의 Datasets은 샘플과 정답을 각각 변경하기 위한 transform과 target_transform을 제공한다.)
)

MNIST_data = datasets.MNIST(
    root = "MINIST",
    train = True,
    download = True,
    transform = ToTensor(),    
)

# 공개 데이터셋에서 테스트 데이터를 내려받습니다.
test_data = datasets.FashionMNIST(
    root="FashionMNIST",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64 # dataloader의 각 객체는 64개의 특징, 정답을 batch의 크기만큼 묶어 반환한다.

# 데이터로더를 생성합니다.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)  # datasets(Tensor)을 dataloader의 인자값으로 전달한다. 이는 iterable로 감싸고, batch, sampling, suffle, 및 
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)       # 다중 프로세스로 데이터 불러오기를 지원한다. batch 크기는 사전에 정의한다.
                                                                                   # DataLoader는 Dataset을 불러온 뒤 필요에 따라 Dataset을 Iterate 할 수 있다.
                                                                                   # DataLoader를 순회하면 Image와 Label을 반환한다. 

for Image, Label in test_dataloader:                                             # DataLoader 순회 후 Train_feature(이미지)와 Train_label(라벨)의 묶음(Batch)을 반환한다.
    print(f"Shape of Image [N, C, H, W]: {Image.shape} {Image.dtype}")           # Image 반환 내용 N(Batch), C(Channel), H(Height), W(width)
    print(f"Shape of Label: {Label.shape} {Label.dtype}")                        # Label 반환 내용 
    break

# 학습에 사용할 CPU나 GPU 장치를 얻습니다.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# 클래스 내에 정의된 함수를 메서드라고 한다. 
# 그리고 메서드의 첫번째 인자는 반드시 self 여야 한다. 
# __init__ 메서드의 첫번째 인자는 self 이지만 호출할 때는 self값을 넘겨주지 않는다. 파이썬이 자동으로 넘겨주기 때문이다. 
# 만약 첫번째 인자를 self로 설정하지 않은경우에 호출을 하면 설정된 인자값이 없으나 1개를 받았다고 에러문구가 나온다. 
# 이것은 파이썬의 메서드는 첫번째 인자로 항상 인스턴스를 넘기기 때문이다. 
# 인스턴스는 클래스의 주소값을 가지고 있고, 이를 직접 할당하여 사용도 가능하다. 
# ex)
# f3 = foo()
# id(f3)  = 47789136
# ->foo.func2(f3)
#  func2(f3) print(id(f3)) -> = 47789136
# f.func2()
# id(f) = 4878554 -> f.func2() = id(self) = 4878554 가 출력 

'''
flatten = numpy에서 지원하는 다차원 배열을 1차원으로 평탄화해주는 함수.
X1 = np.array([[55,11], [22,44], [77,50]])
print("Not Flatten : {}" .format(X1))
X1 = X1.flatten()
print("flatten result : {}" .format(X1))
'''

# 모델을 정의합니다.
class MyModel(nn.Module):                        # nn을 통한 MyModel을 생성.
    def __init__(self):
        super(MyModel, self).__init__()          # 파이썬은 클래스간 상속이 가능하다. super 명령어는 이러한 상속관계에서 부모 클래스를 호출하는 함수이다.
        
        self.flatten = nn.Flatten()              # __init__()에서 신경망의 계층(Layer)을 정의하고 forward에서 신경망에 데이터를 어떻게 전달할지 정한다.
        
        self.linear_relu_stack = nn.Sequential(  # nn.Sequential 클래스는 nn.Linear, nn.ReLU와 같은 모듈을 인수로 받아서 순서대로 정렬하고, 
                                                 #  입력값이 들어오면 순서에 따라 모듈을 실행하여 결과를 반환한다.
                                                 
            nn.Linear(28*28, 512),               # 입력되는 차원의 수와 출력되는 차원의 수를 인자값으로 정한다. 
            nn.ReLU(),                           # torch.nn.Linear(in_features, out_features, bias = True, device = None, dtype = None)
            nn.Linear(512, 512),                 # bias가 false로 설정된 경우 layer는 bias를 학습하지 않는다. 디폴트는 true 
            nn.ReLU(),                           # device는 cpu, gpu를 선택하는 것이다.   
            nn.Linear(512, 10)                   # dtype은 자료형을 선택하는 것이다. 
        )

    def forward(self, x):                       # 신경망에 데이터를 어떻게 전달할지 정한다.
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = MyModel().to(device) # CUDA가 usable할 경우 GPU를 기반으로 학습을 가속화 시키기 위하여 사용한다.
#model = MyModel()


'''
모델 매개변수 최적화하기 
매개변수 = 가중치(weight)를 의미한다. 
모델을 학습 시키기 위하여 손실함수(Loss Function)과 최적화(Optimizer)가 필요하다. 
최적화 : 확률적 경사하강법 사용. SGD - Stochastic Gradient Descent
'''


print("장치는 : {}, 모델은 : {}".format(device, model))

loss_fn = nn.CrossEntropyLoss()                             # 손실 함수 중 다중분류에 적합한 CrossEntropyLoss 함수 이다. 
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)    # 최적화 함수 중 확률적 경사하강법을 사용하여 데이터 수의 영향을 최소화한다.

def train(dataloader, model, loss_fn, optimizer):           # Dataloader: Dataset을 Iterate 하기 위함, Model: MyModel의 규칙, Loss_fn: 손실 함수, Optimizer : 최적화 함수
    size = len(dataloader.dataset)                          # Batch와 DataLoader에 올라간 Dataset의 사이즈를 곱하여 현재 학습이 진행된 양을 산출하는 역할
    
    for batch, (Image, Label) in enumerate(dataloader):             
        Image, Label = Image.to(device), Label.to(device)

        # 예측 오류 계산
        pred = model(Image)
        loss = loss_fn(pred, Label)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(Image)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for Image, Label in dataloader:
            Image, Label = Image.to(device), Label.to(device)
            pred = model(Image)
            test_loss += loss_fn(pred, Label).item()
            correct += (pred.argmax(1) == Label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "mymodel.pth")
print("Saved PyTorch Model State to model.pth")

model = MyModel()
model.load_state_dict(torch.load("mymodel.pth"))

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

model.eval()
x, y = test_data[0][0], test_data[0][1]
print("뭐야이게 ", test_data[0][0])
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
