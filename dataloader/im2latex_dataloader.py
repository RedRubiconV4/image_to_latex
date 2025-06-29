import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from image2Latex.DataLoader.tokeniser import LaTeXTokeniser

class im2latex_dataloader(Dataset):
    def __init__(self, dataset, vocab):
        self.paired_dataset = dataset
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        self.tokeniser = LaTeXTokeniser()
        vocab = pd.read_csv(vocab)
        self.vocab = dict(zip(vocab['Token'], vocab['Index']))
        self.max_length = 200

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, idx):
        paired_sample = self.paired_dataset[idx]
        latex_formula = paired_sample['latex_formula']
        image_path = paired_sample['image_path']

        image = Image.open(image_path).convert('RGB')
        image = image.resize((224,224)) # (256,64), (512,256)
        image = self.transforms(image)

        tokens = self.tokeniser.tokenise(latex_formula)
        tokens = ['<sos>'] + tokens + ['<eos>']

        unk_id = self.vocab.get('<unk>', 3)
        token_ids = [self.vocab.get(tok, unk_id) for tok in tokens]

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        elif len(token_ids) < self.max_length:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))
        token_ids = torch.tensor(token_ids)

        return image, token_ids