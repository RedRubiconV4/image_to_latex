import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import math
import random

class ResNet50_transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=4, nhead=4, dropout=0.0, max_len=200):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # resnet = models.resnet34(pretrained=True) # dim output = 512

        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            # resnet.layer4
        )
        self.conv1 = nn.Conv2d(1024, d_model, kernel_size=1) # 2048, 1024

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def encode(self, x):
        x = self.encoder(x)
        x = self.conv1(x)
        B, C, H, W = x.size()
        return x.view(B, C, H * W).permute(0, 2, 1)

    def add_positional_encoding(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

    def forward(self, image, tgt_seq, teacher_forcing_ratio=0.5): # forward_with_scheduled_sampling
        B, T = tgt_seq.size()
        device = tgt_seq.device
        logits_list = []
        input_seq = tgt_seq[:, 0].unsqueeze(1)

        memory = self.encode(image).permute(1, 0, 2)  # (N, B, d_model)

        for t in range(0, T):
            tgt = self.token_embedding(input_seq)
            tgt = self.add_positional_encoding(tgt)
            tgt = tgt.permute(1, 0, 2)  # (S, B, d_model)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(t+1).to(device)

            out = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.output_projection(out[-1])  # (B, vocab_size)
            logits_list.append(logits.unsqueeze(1))  # Collect output

            # Decide to use ground truth or prediction
            use_teacher = random.random() < teacher_forcing_ratio
            next_input = tgt_seq[:, t] if use_teacher else logits.argmax(dim=-1)
            input_seq = torch.cat([input_seq, next_input.unsqueeze(1)], dim=1)

        return torch.cat(logits_list, dim=1)

    def generate(self, image, max_len=200, sos_token_id=1, eos_token_id=2):
        B = image.size(0)
        device = image.device
        memory = self.encode(image).permute(1, 0, 2)

        ys = torch.ones(B, 1, dtype=torch.long).fill_(sos_token_id).to(device)
        for i in range(max_len):
            tgt = self.token_embedding(ys)
            tgt = self.add_positional_encoding(tgt)
            tgt = tgt.permute(1, 0, 2)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(device)
            out = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
            prob = self.output_projection(out[-1])
            next_token = prob.argmax(dim=-1).unsqueeze(1)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break
        return ys