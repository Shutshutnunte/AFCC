def is_square(indeces, temp_field):
  sum1 = 0
  for i in indeces:
    for j in indeces:
      sum1+= temp_field[i,j]
  if sum1 == len(indeces)**2:
    return True
  return False



def return_squares(labels_one, temp_field):
  temp_labels_one = []
  temp_labels_one += labels_one
  clusters = []
  for lb in labels_one:
    if lb in temp_labels_one:
      cluster = []
      cluster.append(lb)
      for lb2 in temp_labels_one[1:]:
        if is_square(cluster + [lb2],temp_field):
          cluster.append(lb2)
      for c in cluster:
        temp_labels_one.remove(c)
      clusters.append(cluster)
  return clusters

# kernel_size = 4
# num_filters = 512
num_classes = 100
num_filters = out_shape[1]
feature_size = out_shape[2]
values_matrix= torch.zeros(num_filters,num_classes,num_classes).cpu()


fc_model = FC_Net(num_filters = num_filters, feature_size = feature_size).cuda()
fc_model.cuda()

model_name = "vgg16_d64_layer7.pth"
path = F"/content/drive/MyDrive/vggs/CIFAR100_100/{model_name}"
a_state = torch.load(path)
fc_model.load_state_dict(a_state,strict = False)


# Convert to CONV for faster implementation
unflatten_layer = nn.Unflatten(1,(num_filters,feature_size,feature_size))
conv_layer = nn.Conv2d(num_filters,num_classes,feature_size,bias=False)
conv_layer.weight.data = copy.deepcopy(unflatten_layer(fc_model.fc.weight.data))


# Getting the values matrices which will be used to calculate SFP:
for i, (images, labels) in enumerate(test_dataloader):
  if use_cuda:
    images = images.cuda()
    labels = labels.cuda()

  with torch.no_grad():
    between = model2(images)[which_layer]


    outputs = torch.einsum('ijlm,kjlm->jki', conv_layer.weight.data, between)


  index_tensor = labels.unsqueeze(1).expand(100, num_classes)
  for n in range(num_filters):
    values_matrix[n].scatter_add_(0, index_tensor.cpu(), outputs[n,:,:].cpu())


torch.save(values_matrix,"/content/drive/MyDrive/vggs/CIFAR100_100/vgg6_d16_values_matrix_layer7.pth")


# Calculating the SFP's important values:
normalized_values = values_matrix/ 1000 #success_matrix.sum(2).unsqueeze(2)#values_matrix/1000 #values_matrix/ success_matrix.sum(2).unsqueeze(2)

for k in range(num_filters):
  if normalized_values[k,:,:].max()!=0:
    normalized_values[k,:,:] = normalized_values[k,:,:]/normalized_values[k,:,:].max()

th = 0.3
one_and_zeros_values = copy.deepcopy(normalized_values)
one_and_zeros_values[one_and_zeros_values>th] = 1
one_and_zeros_values[one_and_zeros_values<=th] = 0

# Computing clusters:
all_clusters = {}
noise_filter = torch.zeros(num_filters)

all_clusters_with_zeros = {}
noise_filter_with_zeros = torch.zeros(num_filters)

for i in range(num_filters):
  indexes = []
  for j in range(num_classes):
    if one_and_zeros_values[i,j,j] == 1:
      indexes.append(j)
  # random.shuffle(indexes)
  clusters = return_squares(indexes,one_and_zeros_values[i,:,:])


  noise_filter_with_zeros[i] = one_and_zeros_values[i,:,:].sum() -np.sum([len(a_list)**2 for a_list in clusters])
  all_clusters_with_zeros[i] = clusters
  if np.sum([len(a_list)**2 for a_list in clusters]) == 0 :
    print("0")
    noise_filter[i] = -1
    all_clusters[i] = clusters
  else:
    noise_filter[i] = one_and_zeros_values[i,:,:].sum() -np.sum([len(a_list)**2 for a_list in clusters])
    all_clusters[i] = clusters

new_ones_and_zeros = copy.deepcopy(one_and_zeros_values)
noise_cluster = torch.zeros(num_filters)
for i in range(num_filters):
  for a_list in all_clusters[i]:
    for j in a_list:
      for k in a_list:
        new_ones_and_zeros[i,j,k] = 0
    for j in a_list:
      noise_cluster[i] += new_ones_and_zeros[i,:,j].sum()

cluster_sizes = torch.zeros(num_filters)
for i in range(num_filters):
  for a_list in all_clusters[i]:
    cluster_sizes[i] += len(a_list)

#Printing cluster data:
torch.mean(cluster_sizes)
torch.mean(noise_filter_with_zeros)
torch.mean(noise_filter[noise_filter>-1])
all_lengthes = []
all_lengthes_with_zeros = []

for i in range(num_filters):
  if len(all_clusters[i]) >0:
    all_lengthes.append([len(a_list) for a_list in all_clusters[i] if len(a_list)>0])
  all_lengthes_with_zeros.append([len(a_list) for a_list in all_clusters[i] if len(a_list)>0])


all_lengthes_flat = []
all_lengthes_flat_with_zeros = []
for sublist in all_lengthes:
    for item in sublist:
        all_lengthes_flat.append(item)

for sublist in all_lengthes_with_zeros:
    for item in sublist:
        all_lengthes_flat_with_zeros.append(item)

print(np.mean([sum(l) for l in all_lengthes]))
print(np.mean([sum(l) for l in all_lengthes_with_zeros]))
print(torch.mean(torch.tensor([ len(x) for x in all_lengthes]).float()))
print(torch.mean(torch.tensor([ len(x) for x in all_lengthes_with_zeros]).float()))
print(torch.mean(torch.tensor(all_lengthes_flat).float()))
print(torch.mean(torch.tensor(all_lengthes_flat_with_zeros).float()))
all_all_clusters = []
for sublist in all_clusters.values():
    for item in sublist:
        all_all_clusters.append(item)

all_all_all_clusters = []
for sublist in all_all_clusters:
    for item in sublist:
        all_all_all_clusters.append(item)
import matplotlib.pyplot as plt
weights = np.ones_like(all_lengthes_flat) / len(all_lengthes_flat)

plt.hist(all_lengthes_flat)#, weights=weights)
