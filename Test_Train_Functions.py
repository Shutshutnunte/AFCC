# Test accuracy

def test_func(test_dataloader,fc_model,model2 = None,which_layer = None):
  total = 0
  correct_soft = 0
  if model2:
    model2.eval()
  fc_model.eval()
  for i, (images, labels) in enumerate(test_dataloader):
    if use_cuda:
      images = images.cuda()
      labels = labels.cuda()
    with torch.no_grad():
      if model2:
        between = model2(images)[which_layer]
        outputs = fc_model(between)
      else:
        outputs = fc_model(images)
    p_max, predicted_soft = torch.max(outputs, 1)
    correct_soft += (predicted_soft == labels).sum()


    total += labels.size(0)
  return float(correct_soft/total)

def train_func(train_dataloader,optimizer,criterion,fc_model,all_masks,model2 = None,which_layer = None,D=None):
  if model2:
    model2.eval()

  fc_model.train()

  for i, (images, labels) in enumerate(train_dataloader):
    if D:
      images = aug_images(images,D)

    if use_cuda:
      images = images.cuda()
      labels = labels.cuda()

    if model2:
      between = model2(images)[which_layer]
      output = fc_model(between)
    else:
      output = fc_model(images)

    optimizer.zero_grad()
    loss= criterion(output, labels)
    loss.backward()
    optimizer.step()

    for name, param in fc_model.named_parameters():
      if name in all_masks.keys():
        param.data = param.data*all_masks[name]

def reg_train_func(train_dataloader,optimizer,criterion,fc_model,model2 = None,which_layer = None, D=None):
    model2.eval()

    fc_model.train()

    for i, (images, labels) in enumerate(train_dataloader):
        if D:
            images = aug_images(images,D)

        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        if model2:
            between = model2(images)[which_layer]
            output = fc_model(between)
        else:
            output = fc_model(images)

        optimizer.zero_grad()
        loss= criterion(output, labels)
        loss.backward()
        optimizer.step()
