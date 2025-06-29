import os
import math
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dataloader.im2latex_dataloader import im2latex_dataloader as im2latex_dataloader
from dataloader.image_loader import load_image as load_image
from models.resnet_transformer import ResNet50_transformer as ResNet_transformer
from models.vit_transformer import vision_transformer as Vit_transformer
from torch.cuda.amp import autocast, GradScaler

def compute_bleu(preds, gts):
    score = sentence_bleu([gts], preds) # smoothing_function=smooth
    return score

def decode_output(vocab, output):
    unk_id = vocab.get(3, '<unk>')
    words = [vocab.get(ids, unk_id) for ids in output]
    return words

def clean_tokens(tokens, sos=1, eos=2, pad=0):
    cleaned_tokens = []

    for token in tokens:
        if token == sos or token == pad:
            continue
        elif token == eos:
            break
        else:
            cleaned_tokens.append(token)
    return cleaned_tokens

def linear_decay(epoch, total_epochs):
    return max(0, 1.0 - (epoch / total_epochs))

def inverse_sigmoid_decay(epoch, k=10):
    return k / (k + math.exp(epoch / k))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/data_path')
    parser.add_argument('--model', default='resnet_transformer', choices=['resnet_transformer, vision_transformer'])
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--device', default='gpu', choices=['cpu', 'cuda'])
    parser.add_argument('--save_frequency', default=1)
    parser.add_argument('--save_path', default='/save_path')
    parser.add_argument('--checkpoint', default=None)

    parser.add_argument('--lr', default=0.0001)
    parser.add_argument('--batch_size', default=8)
    args = parser.parse_args()

    train_latex_csv = os.path.join(args.data_path, 'im2latex_train_processed.csv')
    val_latex_csv = os.path.join(args.data_path, 'im2latex_validate.csv')
    image_folder = os.path.join(args.data_path, 'formula_image_processed', 'formula_image_processed')
    vocab_csv = os.path.join(args.data_path, 'vocab.csv')
    train_paired_dataset = load_image(train_latex_csv, image_folder)
    train_paired_dataset = train_paired_dataset[0:50000]
    X_train = torch.utils.data.DataLoader(im2latex_dataloader(train_paired_dataset, vocab_csv), batch_size=args.batch_size, shuffle=True)
    val_paired_dataset = load_image(val_latex_csv, image_folder)
    val_paired_dataset = val_paired_dataset[0:5000]
    X_val = torch.utils.data.DataLoader(im2latex_dataloader(val_paired_dataset, vocab_csv), batch_size=args.batch_size, shuffle=True)

    if args.model == 'resnet_transformer':
        model = ResNet_transformer(vocab_size=486)
    elif args.model == 'resnet_transformer':
        model = Vit_transformer(vocab_size=486)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # checkpoint
    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        total_train_loss = checkpoint['train_loss']
        total_train_bleu = checkpoint['train_bleu']
        total_val_loss = checkpoint['val_loss']
        total_val_bleu = checkpoint['val_bleu']
        best_loss = total_val_loss[-1]
        best_bleu = total_val_bleu[-1]
    else:
        start_epoch = 0
        total_train_loss = []
        total_train_bleu = []
        total_val_bleu = []
        total_val_loss = []
        best_loss = 10
        best_bleu = 0.001

    epochs = args.epochs
    best_path = os.path.join(args.save_path, 'model_best_loss.pth')
    best_path2 = os.path.join(args.save_path, 'model_bleu_loss.pth')
    vocab = pd.read_csv(vocab_csv)
    vocab = dict(zip(vocab['Index'], vocab['Token']))
    scaler = GradScaler()

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        train_bleu_score = []
        tf_ratio = inverse_sigmoid_decay(epoch, epochs)

        for image, label in tqdm(X_train):
            image, label = image.to(device), label.to(device)
            optimiser.zero_grad()
            output = model(image, label, tf_ratio)
            # output2 = torch.argmax(output, dim=-1) #for debugging purposes

            preds = model.generate(image)
            preds = preds.tolist()
            labels = label.tolist()
            for i in range(len(labels)):
                pred_sample = clean_tokens(preds[i])
                pred_sample = decode_output(vocab, pred_sample)

                gt_sample = clean_tokens(labels[i])
                gt_sample = decode_output(vocab, gt_sample)

                score = compute_bleu(pred_sample, gt_sample)
                train_bleu_score.append(score)

            output = output.view(-1, 486)
            label = label.view(-1)
            loss = criterion(output, label)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        total_train_loss.append(train_loss / len(X_train))
        print(train_loss / len(X_train))
        total_train_bleu.append(sum(train_bleu_score) / len(X_train))
        print(sum(train_bleu_score) / len(X_train))

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_bleu_score = []

            for image, label in X_val:
                image, label = image.to(device), label.to(device)
                output = model(image, label)

                output_loss = output.view(-1, 486)
                label_loss = label.view(-1)
                loss = criterion(output_loss, label_loss)

                preds = model.generate(image)
                labels = label.tolist()
                preds = preds.tolist()

                for i in range(len(labels)):
                    l = clean_tokens(labels[i])
                    l = decode_output(vocab, l)

                    p = clean_tokens(preds[i])
                    p = decode_output(vocab, p)

                    score = compute_bleu(p, l)
                    val_bleu_score.append(score)
                val_loss += loss.item()

            val_loss_avg = val_loss / len(X_val)
            total_val_loss.append(val_loss_avg)
            print('Validation loss: ' + str(val_loss_avg))
            val_bleu_avg = sum(val_bleu_score) / len(val_bleu_score)
            total_val_bleu.append(val_bleu_avg)
            print('Validation bleu score: ' + str(val_bleu_avg))

        if total_val_loss[-1] < best_loss:
            best_loss = total_val_loss[-1]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'train_bleu': total_train_bleu,
                'train_loss': total_train_loss,
                'val_bleu': total_val_bleu,
                'val_loss': total_val_loss,
            }, best_path)

        save_path = os.path.join(args.save_path, f'model{epoch}.pth')
        if epoch % args.save_frequency == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'train_bleu': total_train_bleu,
                'train_loss': total_train_loss,
                'val_bleu': total_val_bleu,
                'val_loss': total_val_loss,
            }, save_path)
