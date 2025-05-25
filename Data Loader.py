batch_size = 100
num_workers = 8
train_transform = Compose([
    ToTensor(),
    # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = Compose([
    ToTensor(),
    # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Data
train_dataset = CIFAR100(
    root="dataset/", train=True, transform=train_transform, download=True
)
test_dataset = CIFAR100(
    root="dataset/", train=False, transform=test_transform, download=True
)


batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = len(train_dataset), shuffle=True)
x,y = next(iter(train_loader))
x = 2*x-1
# x, y = sort_for_balance(x, y)
# x, y = sort_for_no_overlap(x, y)
train_set = torch.utils.data.TensorDataset(x,y)
# train_dataloader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle=True, num_workers=8)
train_dataloader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = len(test_dataset), shuffle=False)
x,y = next(iter(test_loader))
x = 2*x-1
test_set = torch.utils.data.TensorDataset(x,y)
# test_dataloader = torch.utils.data.DataLoader(dataset = test_set, batch_size = batch_size, shuffle=False, num_workers=8)
test_dataloader = torch.utils.data.DataLoader(dataset = test_set, batch_size = batch_size, shuffle=False)

