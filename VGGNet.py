import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
learning_rate = 0.001

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32), # crop 후 image size = 32x32
    transforms.ToTensor()
  ])

# CIFAR-10 dataset
# ============================================================
# CIFAR-10 데이터셋
# - 32x32 픽셀 크기의 컬러 이미지(RGB 3채널) 60,000장
# - 학습용 50,000장 + 테스트용 10,000장
# - 총 10개 클래스, 클래스당 6,000장 (균등 분포)
#   클래스: 비행기, 자동차, 새, 고양이, 사슴,
#           개, 개구리, 말, 배, 트럭
# - 그래서 마지막 fc 층의 출력이 10 (= 클래스 개수)
#   fc = nn.Linear(1568, 10) 에서 10이 분류할 클래스 수
# - 입력이 3채널인 것도 CIFAR-10이 컬러라서
#   (conv1의 in_channels=3 과 일치)

# 공간 크기(spatial size) 변화 추적   [입력: 32x32, 3채널]
# 패딩 없음 → conv 출력 = (입력 - kernel + 1)
#
#  입력            : 3  x 32 x 32
#  conv1(k=3,s=1)  : 16 x 30 x 30   (32 - 3 + 1 = 30)
#  conv2(k=3,s=1)  : 32 x 28 x 28   (30 - 3 + 1 = 28)
#  maxpool(4)      : 32 x  7 x  7   (28 / 4 = 7, stride 기본값=4) # DON'T FORGET the POOLING!
#  flatten         : 32 * 7 * 7 = 1568  → fc(1568, 10) 입력과 일치
# ============================================================
train_dataset = torchvision.datasets.CIFAR10(root = '../../data',
                                             train = True,
                                             transform = transform,
                                             download = True)

test_dataset = torchvision.datasets.CIFAR10(root = '../../data',
                                            train = False,
                                            transform = transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = 100,
                                           shuffle = True
                                          )

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = 100,
                                          shuffle = False)

class VGG(nn.Module):
  def __init__(self):
    super(VGG, self).__init__()
    self.maxpool = nn.MaxPool2d(4)
    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1)
    self.conv2 = nn.Conv2d(in_channels = 16, out_channels =32, kernel_size = 3, stride = 1)
    self.relu = nn.ReLU(inplace = True)
    self.fc = nn.Linear(1568, 10)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.maxpool(x)
    x = x.view(x.size(0), -1)
    #print(x.shape)
    x = self.fc(x)

    return x

model = VGG().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def update_lr(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

total_step = len(train_loader)
curr_lr = learning_rate

# ── 1 epoch 학습 ──────────────────────────────────────────
def train(epoch):
  global curr_lr
  model.train()                       # 학습 모드
  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()             # 이전 gradient 초기화
    loss.backward()                   # 역전파
    optimizer.step()                  # 가중치 업데이트

    if (i+1) % 100 == 0:
      print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
            epoch, num_epochs, i+1, total_step, loss.item()))

  # 일정 epoch마다 학습률 감소
  if epoch % 20 == 0:
    curr_lr /= 3
    update_lr(optimizer, curr_lr)

# ── 테스트셋 정확도 측정 ──────────────────────────────────
def test():
  model.eval()                        # 평가 모드
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images) # 출력 모양 [배치크기, 10] — 각 이미지마다 10개 클래스 점수(logit)

    # torch.max(텐서, dim=1) → (최댓값, 최댓값의 인덱스) 튜플 반환
    # 점수 자체(최댓값)는 안 쓰니 _ 에 버리고, 예측 클래스 = 인덱스만 사용
    # dim=1 → 클래스 축(10개 방향)을 따라 최댓값 위치를 찾음 → 예측 클래스(0~9)

      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)

    # predicted == labels : 원소별 비교 → True/False 텐서
    # .sum()  : True를 1로 세서 합산 → 이 배치에서 맞춘 개수
    # .item() : 텐서를 파이썬 숫자로 변환해 correct에 누적
      correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# ── 실행 ──────────────────────────────────────────────────
for epoch in range(1, 10):
  train(epoch)
  test()