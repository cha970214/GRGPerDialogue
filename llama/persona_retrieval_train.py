import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
import random
import json
import os

from transformers import BertModel, BertTokenizer

class PUPDataset(Dataset):

    def __init__(self, pup_file: str):
        super().__init__()
        with open(pup_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        

        self.all_samples = self.data

        self.conv_dict = {}
        for item in self.all_samples:
            cid = item["conversation_id"]
            if cid not in self.conv_dict:
                self.conv_dict[cid] = []
            self.conv_dict[cid].append(item)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        sample = self.data[idx]
        c_id = sample["conversation_id"]
        utterance = sample["utterance"]
        pos_persona = sample["persona"]

        neg_persona = None
        while True:
            candidate = random.choice(self.all_samples)

            if (candidate["conversation_id"] != c_id) and (candidate["persona"] != pos_persona):
                neg_persona = candidate["persona"]
                break

        return {
            "conversation_id": c_id,
            "utterance": utterance,
            "pos_persona": pos_persona,
            "neg_persona": neg_persona,
        }


class BiEncoderForRetrieval(nn.Module):
    """
    DPR-style Bi-Encoder:
    - Utterance Encoder: BERT
    - Persona Encoder: BERT
    """
    def __init__(self, model_name_or_path="bert-base-uncased", dropout=0.1):
        super().__init__()

        self.utterance_encoder = BertModel.from_pretrained(model_name_or_path)
        self.persona_encoder = BertModel.from_pretrained(model_name_or_path)
        
        self.dropout = nn.Dropout(dropout)
        
    def encode_utterance(self, input_ids, attention_mask):
        outputs = self.utterance_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)
        return cls_emb

    def encode_persona(self, input_ids, attention_mask):
        outputs = self.persona_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)
        return cls_emb
    
    def forward(
        self,
        utterance_input_ids, utterance_attention_mask,
        pos_persona_input_ids, pos_persona_attention_mask,
        neg_persona_input_ids, neg_persona_attention_mask
    ):
        """
        DPR-style forward:
          1) encode utterance -> u
          2) encode pos_persona -> p+
          3) encode neg_persona -> p-
          4) dot product = similarity(u, p+ / p-)
          5) NLL Loss = - log( e^(u·p+) / ( e^(u·p+) + e^(u·p-) ) )
        """

        u = self.encode_utterance(utterance_input_ids, utterance_attention_mask)   
        p_pos = self.encode_persona(pos_persona_input_ids, pos_persona_attention_mask)
        p_neg = self.encode_persona(neg_persona_input_ids, neg_persona_attention_mask)

        sim_pos = torch.sum(u * p_pos, dim=1)  
        sim_neg = torch.sum(u * p_neg, dim=1)  


        numerator = torch.exp(sim_pos)
        denominator = torch.exp(sim_pos) + torch.exp(sim_neg)
        loss = -torch.log(numerator / (denominator + 1e-8) + 1e-8).mean()

        return loss

def train_bi_encoder(
    train_file: str,
    val_file: str,
    model_name_or_path="bert-base-uncased",
    epochs=40,
    lr=1e-5,
    batch_size=16,
    dropout=0.1,
    max_seq_len=128,
    device="cuda",
):

    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

    train_dataset = PUPDataset(train_file)
    val_dataset = PUPDataset(val_file)

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        collate_fn=lambda x: collate_fn_bi_encoder(x, tokenizer, max_seq_len)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=lambda x: collate_fn_bi_encoder(x, tokenizer, max_seq_len)
    )

    model = BiEncoderForRetrieval(model_name_or_path=model_name_or_path, dropout=dropout)
    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        # -- TRAIN --
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            (utterance_input_ids, utterance_attention_mask,
             pos_persona_input_ids, pos_persona_attention_mask,
             neg_persona_input_ids, neg_persona_attention_mask) = [t.to(device) for t in batch]

            optimizer.zero_grad()
            loss = model(
                utterance_input_ids, utterance_attention_mask,
                pos_persona_input_ids, pos_persona_attention_mask,
                neg_persona_input_ids, neg_persona_attention_mask
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # -- VALIDATION --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                (utterance_input_ids, utterance_attention_mask,
                 pos_persona_input_ids, pos_persona_attention_mask,
                 neg_persona_input_ids, neg_persona_attention_mask) = [t.to(device) for t in batch]

                loss = model(
                    utterance_input_ids, utterance_attention_mask,
                    pos_persona_input_ids, pos_persona_attention_mask,
                    neg_persona_input_ids, neg_persona_attention_mask
                )
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"[Epoch {epoch}/{epochs}] train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

    os.makedirs("bi_encoder_output", exist_ok=True)
    torch.save(model.utterance_encoder.state_dict(), "bi_encoder_output/utterance_encoder.pt")
    torch.save(model.persona_encoder.state_dict(), "bi_encoder_output/persona_encoder.pt")
    print("Utterance encoder saved => bi_encoder_output/utterance_encoder.pt")
    print("Persona encoder saved   => bi_encoder_output/persona_encoder.pt")


def collate_fn_bi_encoder(batch, tokenizer, max_length):

    utterances = [item["utterance"] for item in batch]
    pos_personas = [item["pos_persona"] for item in batch]
    neg_personas = [item["neg_persona"] for item in batch]

    # 1) Encode utterances
    utt_enc = tokenizer(
        utterances, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    # 2) Encode pos_personas
    pos_enc = tokenizer(
        pos_personas, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    # 3) Encode neg_personas
    neg_enc = tokenizer(
        neg_personas, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )

    return (
        utt_enc["input_ids"], utt_enc["attention_mask"],
        pos_enc["input_ids"], pos_enc["attention_mask"],
        neg_enc["input_ids"], neg_enc["attention_mask"],
    )


def main():
    train_file = "pup_session1_train.json"
    val_file = "pup_session1_val.json"

    train_bi_encoder(
        train_file=train_file,
        val_file=val_file,
        model_name_or_path="bert-base-uncased",
        epochs=40,            
        lr=1e-5,              
        batch_size=16,
        dropout=0.1,
        max_seq_len=128,
        device="cuda"
    )

if __name__ == "__main__":
    main()
