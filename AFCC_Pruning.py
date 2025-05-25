# Creating masks based on clusters:
# In Random_AFCC the masks will be randomly generated

mask1 = torch.zeros(fc_model.fc.weight.data.shape).cuda()

all_clusters_layer10 = torch.load("/content/drive/MyDrive/vggs/CIFAR100_100/all_clusters_layer10.pth")
all_clusters_layer9 = torch.load("/content/drive/MyDrive/vggs/CIFAR100_100/all_clusters_layer9.pth")
all_clusters_layer8 = torch.load("/content/drive/MyDrive/vggs/CIFAR100_100/all_clusters_layer8.pth")
all_clusters_layer7 = torch.load("/content/drive/MyDrive/vggs/CIFAR100_100/all_clusters_layer7.pth")

for i in range(num_filters):
  for cluster in all_clusters[i]:
    for j in cluster:
      mask1[j,i*(kernel_size**2):(i+1)*(kernel_size**2)] = 1

all_all_clusters_8 = []
for sublist in all_clusters_layer8.values():
    for item in sublist:
        all_all_clusters_8.append(item)

all_all_all_clusters_8 = []
for sublist in all_all_clusters_8:
    for item in sublist:
        all_all_all_clusters_8.append(item)

all_all_clusters_7 = []
for sublist in all_clusters_layer7.values():
    for item in sublist:
        all_all_clusters_7.append(item)

all_all_all_clusters_7 = []
for sublist in all_all_clusters_7:
    for item in sublist:
        all_all_all_clusters_7.append(item)

mask_layer7 = torch.zeros(my_model.layer8[0].weight.data.shape).cuda()
for i in range(512):
  for ii in range(256):
    for iteam in all_all_clusters_8[i]:
        if iteam in all_all_clusters_7[ii]:
          mask_layer7[i,ii,:,:] = 1

optimizer = optim.SGD([
    {
        "params": chain(my_model.layer1.parameters(),my_model.layer2.parameters(),my_model.layer3.parameters(),
                        my_model.layer4.parameters(),my_model.layer5.parameters(),my_model.layer6.parameters(),
                        #my_model.layer7.parameters()#,my_model.layer8.parameters()
                        ),#model.stem.parameters()),#, model.blocks.parameters()),
        "lr": 0.0,#lr * 0.1,
        "momentum": 0.0,
        "weight_decay": 0e-3
    },
    {
        "params": chain(my_model.layer7.parameters(),my_model.layer8.parameters()),
        "lr": 0.01, #* 0.2,
        "momentum": 0.95,
    },
    {
        "params": chain(my_model.layer9.parameters(),my_model.layer10.parameters()),
        "lr": 0.0, #* 0.2,
        "momentum": 0.0,
        "weight_decay": 0e-3
    },
    {
        "params": my_model.fc3.parameters(),
        "lr":0.0,
        "momentum": 0.0,
        "weight_decay": 0e-3
    }],
    momentum=0.9, weight_decay=1e-3, nesterov=True)



my_model.layer8[0].weight.data = my_model.layer8[0].weight.data*mask_layer7

path = "/content/drive/MyDrive/vggs/CIFAR100_100/vgg16_CIFAR100.pth"
a_state = torch.load(path)
my_model.load_state_dict(a_state,strict = False)

model_name = "vgg16_d64_layer10.pth"
path = F"/content/drive/MyDrive/vggs/CIFAR100_100/{model_name}"
a_state = torch.load(path)
my_model.fc3[0].weight.data = copy.deepcopy(a_state['fc.weight'])

test_accuracy_soft = np.zeros(num_epochs)
criterion = nn.CrossEntropyLoss() # MSELoss()

num_epochs = 200

optimizer = optim.SGD(fc_model.parameters(),lr= 0.01 , momentum=0.975 , weight_decay =  0.001, nesterov=True)

for epoch in range(num_epochs):
  if epoch%1==0:
    optimizer.param_groups[0]['lr'] *=0.975

  train_func(train_dataloader,optimizer,criterion,fc_model = my_model ,model2 = None,which_layer = None,D=D)


  # Test accuracy
  acc = test_func(test_dataloader,fc_model= my_model,model2 = None,which_layer = None)
  print(acc)
  test_accuracy_soft[epoch] = acc
