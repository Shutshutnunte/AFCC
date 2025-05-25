# Cutting the VGG architecutre in order to extrapolate the SFP matrices
d = 64
# path = "/content/drive/MyDrive/vggs/CIFAR100_100/vgg16_d64_layer10_AFCC.pth"#F"/content/drive/MyDrive/vggs/CIFAR10/diff d/vgg16_d{d}.pth"
# a_state = torch.load(path)
# my_model.load_state_dict(a_state)


path = "/content/drive/MyDrive/vggs/CIFAR100_100/vgg16_CIFAR100.pth"
a_state = torch.load(path)
my_model.load_state_dict(a_state,strict = False)
# path = "/content/drive/MyDrive/vggs/CIFAR100_100/vgg16_d64_layer10.pth"#"/content/drive/MyDrive/vggs/CIFAR100_100/vgg16_d64_layer10_AFCC.pth"#F"/content/drive/MyDrive/vggs/CIFAR10/diff d/vgg16_d{d}.pth"
# a_state = torch.load(path)

# my_model.fc3[0].weight.data = a_state["fc.weight"]



which_layer = "layer7"
from torchvision.models.feature_extraction import create_feature_extractor
return_nodes = {
    which_layer: which_layer
}
model2 = create_feature_extractor(my_model, return_nodes=return_nodes).cuda()
model2.eval()


class FC_Net(nn.Module):
  def __init__(self,num_filters,feature_size):
    super(FC_Net, self).__init__()
    # Define stem:
    # self.avg_pool = nn.AvgPool2d(7)
    self.feature_size = feature_size
    self.num_filters = num_filters
    self.fc =  nn.Linear((self.feature_size**2)*self.num_filters, 100, bias = False)
    init_range = 1.0 / math.sqrt(self.fc.weight.shape[1])
    nn.init.uniform_(self.fc.weight, a=-init_range, b=init_range)

  def forward(self, x):
    x = x.reshape(x.shape[0], -1)
    y = self.fc(x)
    return y

out_shape = model2(torch.randn(1,3,32,32).cuda())[which_layer].shape
num_filters = out_shape[1]
feature_size = out_shape[2]
fc_model = FC_Net(num_filters=num_filters,feature_size=feature_size).cuda()
fc_model.cuda()

import copy

# fc_model.fc.weight.data = copy.deepcopy(a_state['fc.weight'])
# fc_model.eval()
