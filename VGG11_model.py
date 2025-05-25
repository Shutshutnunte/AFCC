use_cuda = torch.cuda.is_available()
print('Use GPU?', use_cuda)

stam_num = 1
num_train = 50000

# Define a VGG-16

class model(nn.Module):
    def __init__(self, num_classes=100):
        super(model, self).__init__()
        self.num_filters = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, self.num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.num_filters, self.num_filters*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_filters*2),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(self.num_filters*2,self.num_filters*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_filters*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(self.num_filters*2, self.num_filters*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_filters*4),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(self.num_filters*4, self.num_filters*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_filters*4),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(self.num_filters*4, self.num_filters*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_filters*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(self.num_filters*4, self.num_filters*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_filters*8),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(self.num_filters*8, self.num_filters*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_filters*8),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(self.num_filters*8, self.num_filters*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_filters*8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        # self.layer11 = nn.Sequential(
        #     nn.Conv2d(self.num_filters*8, self.num_filters*8, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.num_filters*8),
        #     nn.ReLU())
        # self.layer12 = nn.Sequential(
        #     nn.Conv2d(self.num_filters*8, self.num_filters*8, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.num_filters*8),
        #     nn.ReLU())
        # self.layer13 = nn.Sequential(
        #     nn.Conv2d(self.num_filters*8, self.num_filters*8, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.num_filters*8),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = 2, stride = 2))
        # self.fc= nn.Sequential(
        #         nn.Linear(self.num_filters*8 , 4096))#*4*4*4*2#num_classes
        # self.fc1= nn.Sequential(
        #         nn.Linear(4096, 4096))#*4*4*4*2#num_classes
        # self.fc2= nn.Sequential(
        #         nn.Linear(4096,num_classes, bias = False))#*4*4*4*2#num_classes
        self.fc3= nn.Sequential(
                nn.Linear(self.num_filters*8*4 , 100,bias = False))#*4*4*4*2#num_classes
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        # out = self.layer11(out)
        # out = self.layer12(out)
        # out = self.layer13(out)
        # out = self.max_pool2(out)
        out = out.reshape(out.size(0), -1)
        # out = self.fc(out)
        # out = self.fc1(out)
        # out = self.fc2(out)
        out = self.fc3(out)
        return out
my_model = model()

if use_cuda:
  my_model = my_model.cuda()  # transfer model to GPU


summary(my_model,(3,32,32))
num_epochs = 200
minibatch_size = 100 # 128
criterion = nn.CrossEntropyLoss() # MSELoss()

# optimizer = optim.SGD(my_model.parameters(), lr=0.01, momentum=0.95, weight_decay = 0.001, nesterov=True)
