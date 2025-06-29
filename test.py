import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dataloader.im2latex_dataloader import im2latex_dataloader as im2latex_dataloader
from dataloader.image_loader import load_image as load_image
from models.resnet_transformer import ResNet50_transformer as ResNet_transformer
from models.vit_transformer import vision_transformer as Vit_transformer

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/data_path')
    parser.add_argument('--model', default='resnet_transformer', choices=['resnet_transformer, vision_transformer'])
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--device', default='gpu', choices=['cpu', 'cuda'])
    parser.add_argument('--save_Frequency', default=1)
    parser.add_argument('--save_path', default='/save_path')
    parser.add_argument('--checkpoint', default=None)

    parser.add_argument('--lr', default=0.0001)
    parser.add_argument('--batch_size', default=1)
    args = parser.parse_args()

    test_latex_csv = os.path.join(args.data_path, 'im2latex_test.csv')
    image_folder = os.path.join(args.data_path, 'formula_image_processed', 'formula_image_processed')
    vocab_csv = os.path.join(args.data_path, 'vocab.csv')
    test_paired_dataset = load_image(test_latex_csv, image_folder)
    X_test = torch.utils.data.Dataloader(im2latex_dataloader(test_paired_dataset, vocab_csv), batch_size=args.batch_size)

    if args.model == 'resnet_transformer':
        model = ResNet_transformer(vocab_size=486)
    elif args.model == 'resnet_transformer':
        model = Vit_transformer(vocab_size=486)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # checkpoint
    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        loss = checkpoint['loss']
        loss_array = checkpoint['loss_array']
    model = model.to(device)
    model.eval()

    vocab = pd.read_csv(vocab_csv)
    vocab = dict(zip(vocab['Index'], vocab['Token']))
    preds = []
    gts = []

    for image, label in tqdm(X_test):
        image = image.to(device)
        output = model.generate(image)

        label = label.squeeze(0).tolist()
        label = clean_tokens(label)
        label = decode_output(vocab, label)
        output = output.squeeze(0).tolist()
        output = clean_tokens(output)
        output = decode_output(vocab, output)

        preds.append(output)
        gts.append(label)

    bleu_scores = []
    for pred, gt in zip(preds, gts):
        score = compute_bleu(pred, gt)
        bleu_scores.append(score)
    total_bleu = sum(bleu_scores) / len(bleu_scores)
    print(total_bleu)