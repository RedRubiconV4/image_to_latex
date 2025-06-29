import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
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

    vocab_csv = os.path.join(args.data_path, 'vocab.csv')

    transforms = transforms.Compose([transforms.ToTensor()])
    image = Image.open(args.data_path).convert('RGB')
    image = image.resize((224, 224))
    image = transforms(image)

    if args.model == 'resnet_transformer':
        model = ResNet_transformer(vocab_size=486)
    elif args.model == 'resnet_transformer':
        model = Vit_transformer(vocab_size=486)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # checkpoint
    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    vocab = pd.read_csv(vocab_csv)
    vocab = dict(zip(vocab['Index'], vocab['Token']))

    image = image.to(device)
    output = model.generate(image)

    output = output.squeeze(0).tolist()
    output = clean_tokens(output)
    output = decode_output(vocab, output)

    latex_formula_output = ''.join(output)
    print(latex_formula_output)