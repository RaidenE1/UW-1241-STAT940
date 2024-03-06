import torch
import os
import math
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
from model import (
    TrainDataset,
    ValDataset,
    TestDataset,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152
)
from utils import (
    load_labels,
    get_logger,
    get_seperated_dataset
)
import pandas as pd

TRAIN_DIR = "./data/train/train"
TEST_DIR = "./data/test/test"

# compute by utils/get_mean_std.py
train_mean, train_std = (0.43916297, 0.42551237, 0.40391925), (0.26868406, 0.27018538, 0.2759704)
test_mean, test_std = (0.43916297, 0.42551237, 0.40391925), (0.26868406, 0.27018538, 0.2759704)
# data augmentation
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ]
)

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(test_mean, test_std),
])

# validate
def validate(model, epoch, criterion, val_loader, logger, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # logger.info(f'Validating - Epoch: {epoch + 1}, Batch: {batch_idx}/{len(val_loader)}')

    acc = 100.*correct/total
    logger.info(f'Validating - Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}, acc: {acc}')


def main():
    # parse args
    parser = argparse.ArgumentParser(description='stat940 data challenge 1')
    parser.add_argument('--model', type=str, default="resnet18", metavar='M',
                        help='model to use (default: ResNet18)')
    parser.add_argument('--work-dir', type=str, metavar='WD',
                        help='working dictionary')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--num-classes', type=int, default=18, metavar='NC',
                        help='output classes (default: 10)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--opt', type=str, default="SGD", metavar='OPT',
                        help='optimizer (default: SGD)')
    parser.add_argument('--seed', type=int, default=3407, metavar='S',
                        help='random seed (default: 3407)')
    args = parser.parse_args()
    # hyper parameters setting
    torch.manual_seed(args.seed)
    num_classes = args.num_classes
    num_epochs = args.epochs

    device = torch.device("cuda")
    
    labels = load_labels()
    model_name = "%s_e%d_b%d_lr%d_%s" % (args.model, args.epochs, args.batch_size, args.lr * 1000, args.opt)
    work_dir = os.path.join("./out", args.model, model_name)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    train_log = os.path.join(work_dir, "train.log")
    val_log = os.path.join(work_dir, "val.log")
    with open(train_log, 'w') as _:
        pass
    with open(val_log, 'w') as _:
        pass
    train_logger = get_logger(train_log, name="train")
    val_logger = get_logger(val_log, name="val")
    
    # train_images, val_images = get_seperated_dataset(TRAIN_DIR)
    train_images = os.listdir(TRAIN_DIR)

    train_dataset = TrainDataset(TRAIN_DIR, train_images, transform_train, labels=labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    # val_dataset = ValDataset(TRAIN_DIR, val_images, transform=transform_val, labels=labels)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    if args.model == "resnet18":
        model = ResNet18(num_classes)
    elif args.model == "resnet34":
        model = ResNet34(num_classes)
    elif args.model == "resnet50":
        model = ResNet50(num_classes)
    elif args.model == "resnet101":
        model = ResNet101(num_classes)
    elif args.model == "resnet152":
        model = ResNet152(num_classes)
    else:
        raise NotImplementedError("Model %s not implemented" % (args.model))
    model = model.to(device)
    # compute iteration and warmup iteration
    iterations = math.floor(50000 / args.batch_size)
    warmup_epochs = args.epochs / 100
    warmup_iterations = max(warmup_epochs * iterations, 1000)
    

    criterion = torch.nn.CrossEntropyLoss()
    # choose optimzier
    if args.opt == "SGD":
        initial_lr = args.lr * 0.1
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
    elif args.opt == "Adam":
        initial_lr = args.lr * 0.01
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        raise ValueError("No such optimizer: %s" % (args.opt))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) 
    # start training
    for epoch in range(num_epochs):
        acc = 0
        total = 0
        model.train()
        epoch_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # warmup
            current_iterations = epoch * iterations + batch_idx + 1
            if current_iterations < warmup_iterations:
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = initial_lr + (args.lr - initial_lr) * current_iterations / warmup_iterations

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            acc += predicted.eq(labels).sum().item()

            loss.backward()
            optimizer.step()
        train_logger.info(f'Training - Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}, Acc: {acc / total * 100}')
        # if epoch > args.epochs * 0.8:
        #     validate(model, epoch, criterion, val_loader, val_logger, device)
        # else:
        #     if (epoch + 1) % 5 == 0:
        #         validate(model, epoch, criterion, val_loader, val_logger, device)
        scheduler.step()
        
    # testing
    model.eval()
    test_dataset = TestDataset(TEST_DIR, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    id = []
    label = []
    batch_count = 0
    with torch.no_grad():
        for inputs, image_name in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            predicted = predicted.to(torch.device('cpu'))
            for idx in range(len(image_name)):
                id.append(image_name[idx])
                label.append(int(predicted[idx]))
            batch_count += 1
            train_logger.info(f'Testing - Batch: {batch_count}/{len(test_loader)}')
    # write result to csv
    df = pd.DataFrame({'id': id, 'label': label})
    df.to_csv('./out/%s/%s/%s.csv' % (args.model, model_name, model_name), index=False)

if __name__ == '__main__':
    main()