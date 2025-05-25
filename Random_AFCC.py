# Performing the random permutations by creating random clusters for each filter, while ensuring homogenous distribution of labels among clusters.
filters_num = 512
permutations_size = 5
classes_num = 100
from random import sample
def get_permutations_matrix2(filters_num, permutations_size, classes_num):
  whole_cycles_num = (filters_num * permutations_size) // classes_num
  permutations_matrix = -torch.ones(filters_num, permutations_size)
  available_filters = list(range(filters_num))
  next_available_locations = torch.zeros(filters_num).type(torch.long)
  classes = set(range(classes_num))
  for i in range(whole_cycles_num):
    for label in classes:
      counter = 0
      while True:
        row = sample(available_filters, 1)[0]
        if not label in permutations_matrix[row, :].tolist():
          break
        counter += 1
        if counter == 1000:
          # print(permutations_matrix)
          raise Exception(f'Could not find a place for label {label} after {counter} tries')
      column = next_available_locations[row]
      permutations_matrix[row, column] = label
      next_available_locations[row] += 1
      if next_available_locations[row] == permutations_size:
        available_filters.remove(row)

  for filter in available_filters:
    matrix_row = permutations_matrix[filter, :]
    taken_values_num = next_available_locations[filter]
    taken_values = matrix_row[:taken_values_num].tolist()
    available_labels = classes - set(taken_values)
    available_labels = list(available_labels)
    permutations_matrix[filter, taken_values_num:] = torch.Tensor(sample(available_labels, permutations_size - taken_values_num))
  return permutations_matrix


permutations_matrix = get_permutations_matrix2(filters_num, permutations_size, classes_num)

def initialize_weights_AFCC(my_model, mask):
    # inputs_number_vector = mask.sum(dim=1)
    for i in range(mask.shape[0]):
      current_input_indices = mask[i, :].nonzero()
      inputs_number = current_input_indices.shape[0]
      k = 1 / inputs_number
      k = k ** 0.5
      my_model.fc3[0].weight.data[i, current_input_indices] = (torch.rand(inputs_number, 1).cuda() - 1 / 2) * 2 * k

criterion = nn.CrossEntropyLoss() # MSELoss()
my_model = model()

if use_cuda:
  my_model = my_model.cuda()  # transfer model to GPU

initialize_weights_AFCC(my_model,all_masks_train['fc3.0.weight'])

num_epochs = 260
test_accuracy_soft = np.zeros(num_epochs)

# optimizer = optim.SGD(my_model.parameters(),lr=0.002, momentum=0.975, weight_decay = 0.004, nesterov=True)

optimizer = optim.SGD([{
        "params": chain(my_model.layer1.parameters(),my_model.layer2.parameters(),my_model.layer3.parameters(),
                        my_model.layer4.parameters(),my_model.layer5.parameters(),my_model.layer6.parameters(),
                        my_model.layer7.parameters(),my_model.layer8.parameters(),my_model.layer9.parameters(),
                        my_model.layer10.parameters()),
        "lr": 0.028 ,
        "momentum": 0.9175,
        "weight_decay": 1e-3,
    },
    {
        "params": my_model.fc3.parameters(),
        "lr": 0.01 ,
        "momentum": 0.9175,
        "weight_decay": 1e-3,
    }],
    momentum=0.9, weight_decay=1e-3, nesterov=True)

for epoch in range(num_epochs):
  if epoch%20==0 and epoch!=0:
    optimizer.param_groups[0]['lr'] *=0.6

  train_func(train_dataloader,optimizer,criterion,fc_model = my_model,all_masks = all_masks_train ,model2 = None,D=D)#,mask_layer8=mask_layer8,mask_layer9=mask_layer9,mask_layer10=mask_layer10



  # for name, param in my_model.named_parameters():
  #   try:
  #     param = param*all_masks_train[name]
  #   except:
  #     pass

  # my_model.layer8[0].weight.data = my_model.layer8[0].weight.data*mask_layer7
  # my_model.layer10[0].weight.data = my_model.layer10[0].weight.data*mask_layer9
  # my_model.layer9[0].weight.data = my_model.layer9[0].weight.data*mask_layer9_silnce8



  # Test accuracy
  acc = test_func(test_dataloader,fc_model= my_model,model2 = None)
  print(acc)
  test_accuracy_soft[epoch] = acc

  if epoch == 200:
    torch.save(my_model.state_dict(),"/content/drive/MyDrive/vggs/CIFAR100_100/vgg16_ofek_vgg_epoch200.pth")
