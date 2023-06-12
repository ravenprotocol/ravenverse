import torch
import torch.nn as nn

class Net(nn.Module):    
    def __init__(self):
        super(Net, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding='same')
        self.act_1 = nn.ReLU()
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.drp_1 = nn.Dropout(0.25)
        self.bn_1 = nn.BatchNorm2d(16)
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding='same')
        self.act_2 = nn.ReLU()
        self.maxpool2d_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.drp_2 = nn.Dropout(0.25)
        self.bn_2 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(in_features=32,out_features=256)
        self.act_3 = nn.ReLU()
        self.drp_3 = nn.Dropout(0.4)
        self.bn_3 = nn.BatchNorm1d(256)
        self.dense_2 = nn.Linear(in_features=256, out_features=10)
        self.act_4 = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.conv2d_1(x)
        out = self.act_1(out)
        out = self.maxpool2d_1(out)
        out = self.drp_1(out)
        out = self.bn_1(out)
        out = self.maxpool2d_2(out)
        out = self.conv2d_2(out)
        out = self.act_2(out)
        out = self.maxpool2d_3(out)
        out = self.drp_2(out)
        out = self.bn_2(out)
        out = self.flatten(out)
        out = self.dense_1(out)
        out = self.act_3(out)
        out = self.drp_3(out)
        out = self.bn_3(out)
        out = self.dense_2(out)
        out = self.act_4(out)
        return out

model = Net()

model_script = torch.jit.script(model)

model_script.save('test_model.pt')
