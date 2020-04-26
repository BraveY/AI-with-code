from train import *




def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 2
    net = MyCNN()
    train_path = C.train_path
    test_path = C.test_path
    tensorboard_path = C.tensorboard_path

    train_ds = MyDataset(train_path)
    new_train_ds, validate_ds = dataset_split(train_ds, 0.8)
    test_ds = MyDataset(test_path, train=False)

    train_loader = dataloader(train_ds)
    new_train_loader = dataloader(new_train_ds)
    validate_loader = dataloader(validate_ds)
    test_loader = dataloader(test_ds)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(epochs,new_train_loader,device,net,criterion,optimizer,tensorboard_path)
    validate(validate_loader, device, net, criterion)
    # print("validate acc:",validate(validate_loader,device,net,criterion))
    submission(csv_path=C.csv_path,test_loader=test_loader,device=device,model=net)
    torch.save(net.state_dict(), C.model_save_path)



if __name__ == '__main__':
    main()
