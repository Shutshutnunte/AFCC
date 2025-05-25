#  Code for training the network:
test_accuracy_soft = np.zeros(num_epochs)
criterion = nn.CrossEntropyLoss() # MSELoss()

num_epochs = 200

optimizer = optim.SGD(fc_model.parameters(),lr= 0.01 , momentum=0.975 , weight_decay =  0.001, nesterov=True)

for epoch in range(num_epochs):
  if epoch%1==0:
    optimizer.param_groups[0]['lr'] *=0.975

  train_func(train_dataloader,optimizer,criterion,fc_model,model2 = model2,which_layer = which_layer, D=D)


  # Test accuracy
  acc = test_func(test_dataloader,fc_model,model2 = model2, which_layer = which_layer)
  print(f"epoch: {epoch},  acc: {acc}")
  test_accuracy_soft[epoch] = acc
