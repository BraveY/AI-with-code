import torch.nn.functional as F
from torch import nn

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)  # 按照公式计算后经过卷积层不改变尺寸
        self.pool = nn.MaxPool2d(2, 2)  # 2*2的池化 池化后size 减半
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 56 * 56, 256)  # 两个池化，所以是224/2/2=56
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    #         self.dp = nn.Dropout(p=0.5)
    def forward(self, x):
        #         print("input:", x)
        x = self.pool(F.relu(self.conv1(x)))
        #         print("first conv:", x)
        x = self.pool(F.relu(self.conv2(x)))
        #         print("second conv:", x)

        x = x.view(-1, 16 * 56 * 56)  # 将数据平整为一维的
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #         x = F.log_softmax(x,dim=1) NLLLoss()才需要，交叉熵不需要
        return x

