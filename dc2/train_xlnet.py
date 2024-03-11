import torch
import os
from torch import nn
import numpy as np
from utils.dataset import (
    CustomDataset,
    TestDataset,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.logger import get_logger
from torch import optim
from model.bert_classifier import BertClassifier
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.load_data import *
import argparse
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sep_data():
    data = list(load_data().keys())
    np.random.shuffle(data)
    train_data = data[:int(len(data) * 0.75)]
    val_data = data[int(len(data) * 0.75):]
    return train_data, val_data


def main():
    # parse args
    parser = argparse.ArgumentParser(description='stat940 data challenge 1')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--optim', type=str, default="Adam", metavar='OPT',
                        help='optimizer (default: Adam)')
    parser.add_argument('--seed', type=int, default=3407, metavar='S',
                        help='random seed (default: 3407)')
    parser.add_argument('--val', action='store_true', help='Set the validation to True')
    args = parser.parse_args()

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained("xlnet-large-cased")
    model = AutoModelForSequenceClassification.from_pretrained("xlnet-large-cased", num_labels=3)

    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    
    torch.distributed.init_process_group(backend='nccl')
    
    model.to(device)

    data_labels = load_data()
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    
    if args.local_rank == 0:
        # prepare logging
        if args.val:
            model_name = "%s_e%d_b%d_lr%d_%s_token512" % ("xlnet_val", \
                epochs, batch_size, learning_rate * 10000000, args.optim)
            work_dir = os.path.join("./out", "xlnet_val", model_name)
        else:
            model_name = "%s_e%d_b%d_lr%d_%s_token512" % ("xlnet", \
                epochs, batch_size, learning_rate * 10000000, args.optim)
            work_dir = os.path.join("./out", "xlnet", model_name)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        train_log = os.path.join(work_dir, "train.log")
        with open(train_log, 'w') as _:
            pass
        train_logger = get_logger(train_log, name="train")
        if not args.val:
            val_log = os.path.join(work_dir, "val.log")
            with open(val_log, 'w') as _:
                pass
            val_logger = get_logger(val_log, name="val")
    
    # prepare train dataset
    if not args.val:
        train_data, val_data = sep_data()
    else:
        train_data = list(load_data().keys())
    train_dataset = CustomDataset(train_data, tokenizer)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=8)
    if not args.val:
        # prepare val dataset
        val_dataset = CustomDataset(val_data, tokenizer)
        # val_sampler = DistributedSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=8)
    
    if args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif args.optim == "SGD":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-6)
    else:
        raise ValueError("No such optimizer: %s" % (args.optim))
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) 
    # train&val
    for epoch in range(epochs):
        model.train()
        train_acc = 0
        train_loss = 0
        batch_idx = 0
        for train_input, labels in train_loader:
        # for data in train_data:
        #     label = data_labels[data]
        #     text = tokenizer(
        #         data,
        #         padding = "max_length",
        #         max_length = 512,
        #         truncation = True,
        #         return_tensors = "pt"
        #     )
            optimizer.zero_grad()
            input_ids = train_input['input_ids'].squeeze(1).to(device)
            attention_masks = train_input['attention_mask'].to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_masks=attention_masks).logits
            batch_loss = criterion(output, labels)
            train_loss += batch_loss.item()
            acc = (output.argmax(dim=1) == labels).sum().item()
            # print(batch_loss, acc)
            train_acc += acc
            batch_loss.backward()
            optimizer.step()
        scheduler.step()
        if args.local_rank == 0:
            train_logger.info(f'Training - Epoch {epoch + 1}, LR: {optimizer.param_groups[0]["lr"]}, Loss: {4 * train_loss / len(train_data):.4f}, Acc: {4 * train_acc / len(train_data):.4f}')
        if not args.val:
            # validate
            if (epoch + 1) % 2 == 0:
                val_acc = 0
                val_loss = 0
                model.eval()
                with torch.no_grad():
                    for val_input, val_label in val_loader:
                        val_label = val_label.to(device)
                        input_ids = val_input['input_ids'].squeeze(1).to(device)
                        token_type_ids = val_input['token_type_ids'].to(device)
                        attention_masks = val_input['attention_mask'].to(device)
                        labels = labels.to(device)
                        output = model(input_ids=input_ids, attention_masks=attention_masks).logits

                        batch_loss = criterion(output, val_label)
                        val_loss += batch_loss.item()

                        acc = (output.argmax(dim=1) == val_label).sum().item()
                        val_acc += acc
                if args.local_rank == 0:
                    val_logger.info(f'Validating - Epoch {epoch + 1}, Loss: {val_loss/len(val_data):.4f}, Acc: {val_acc / len(val_data):.4f}')

    if args.local_rank == 0:
        #test
        test_label = load_test_labels()
        test_dataset = TestDataset(test_label, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        model.eval()
        dic = {}
        with torch.no_grad():
            for ids, texts in test_loader:
                texts = texts.to(device)
                input_ids = texts['input_ids'].squeeze(1).to(device)
                token_type_ids = texts['token_type_ids'].to(device)
                attention_masks = texts['attention_mask'].to(device)
                labels = labels.to(device)
                outputs = model(input_ids=input_ids, attention_masks=attention_masks).logits
                _, predicted = outputs.max(1)
                predicted.to(torch.device('cpu'))
                for idx in range(len(ids)):
                    dic[int(ids[idx])] = int(predicted[idx]) + 1
        ids = list(dic.keys())
        ids.sort()
        stars = [dic[ID] for ID in ids]
        df = pd.DataFrame({'ID': ids, 'stars': stars})
        if args.val:
            df.to_csv('./out/%s/%s/%s.csv' % ("xlnet_val", model_name, model_name), index=False)
        else:
            df.to_csv('./out/%s/%s/%s.csv' % ("xlnet", model_name, model_name), index=False)
if __name__ == '__main__':
    main()