def aug_images(images,D):
  # Generate random transformation indices
  rights = torch.randint(-4, 5, (100,))
  ups = torch.randint(-4, 5, (100,))
  with_flips = torch.randint(0, 2, (100,))

  tensor_tuple_indices = torch.stack([rights, ups, with_flips], dim=1)
  tuple_indices = [tuple(indices.tolist()) for indices in tensor_tuple_indices]




  # Generate index matrix for each transformation
  ind_mat_list = [D[a_tuple] for a_tuple in tuple_indices]
  ind_mat = torch.stack(ind_mat_list)
  ind_mat = ind_mat.reshape(ind_mat.shape[0],1,ind_mat.shape[1]*ind_mat.shape[2]).repeat(1,3,1)

  # Apply transformations to images
  images = images.view(images.shape[0], images.shape[1], -1)
  images = torch.gather(images, dim = 2 , index = ind_mat.long())


  images[ind_mat==0] = 0
  images = images.reshape(images.shape[0] ,3,32,32)


  mask = (tensor_tuple_indices[:, 0] >= 0) & (tensor_tuple_indices[:, 1] <= 0)
  second_mask = mask*(tensor_tuple_indices[:, 2]>0)
  third_mask = mask*(tensor_tuple_indices[:, 2]==0)
  # if tensor_tuple_indices[mask, 2]:
  if second_mask.nelement() != 0:
    images[second_mask, :, -tensor_tuple_indices[second_mask, 2].int(), (31-tensor_tuple_indices[second_mask, 0]).int()] = images[second_mask, :, 0, 0]
  # else:
  if third_mask.nelement() != 0:
    images[second_mask, :, -tensor_tuple_indices[second_mask, 2].int(), tensor_tuple_indices[second_mask, 0].int()] = images[second_mask, :, 0, 0]

  return images

#We use D as a global parameter, take note:
indexes = torch.reshape(torch.arange(32*32), shape=(1,32, 32))
D = {}
right_range = 4
up_range = 4
for right in [-4,-3,-2,-1,0,1,2,3,4]:
  for down in [-4,-3,-2,-1,0,1,2,3,4]:
    for with_flip in [True,False]:
      aug_indexes = transforms.functional.affine(indexes, 0, (right, down), 1, 0, fill=(0))
      if with_flip:
        aug_indexes = transforms.functional.hflip(aug_indexes)
      D[(right,-down,with_flip)] = aug_indexes.squeeze(0)


