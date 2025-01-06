import json
import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BiEncoderForRetrieval(nn.Module):
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

def load_bi_encoder_model(
    utterance_encoder_path: str,
    persona_encoder_path: str,
    model_name_or_path="bert-base-uncased",
    dropout=0.1,
    device="cuda"
):
    model = BiEncoderForRetrieval(model_name_or_path=model_name_or_path, dropout=dropout)
    model.utterance_encoder.load_state_dict(torch.load(utterance_encoder_path, map_location=device))
    model.persona_encoder.load_state_dict(torch.load(persona_encoder_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def encode_texts_with_utterance_encoder(model, tokenizer, texts, max_length=128, device="cuda"):
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        embeddings = model.encode_utterance(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return embeddings

def encode_texts_with_persona_encoder(model, tokenizer, texts, max_length=128, device="cuda"):
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        embeddings = model.encode_persona(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return embeddings

def select_personas_by_threshold(utt_embed, persona_embeds, persona_texts, threshold=0.68):
    scores = (utt_embed.unsqueeze(0) * persona_embeds).sum(dim=1)  # [N]
    selected = []
    for i, s in enumerate(scores):
        if s.item() >= threshold:
            selected.append(persona_texts[i])
    return selected

def remove_speaker_info(persona_line: str) -> str:
    result = re.sub(r"(?i)speaker\s*[12]", "", persona_line)

    result = result.strip()
    return result

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = load_bi_encoder_model(
        utterance_encoder_path="bi_encoder_output/utterance_encoder.pt",
        persona_encoder_path="bi_encoder_output/persona_encoder.pt",
        model_name_or_path="bert-base-uncased",
        dropout=0.1,
        device=device
    )
    session_file = "session2.txt"
    with open(session_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    persona_list = data.get("persona_list", [])
    current_utterances = data.get("current", [])
    filtered_persona_list = [remove_speaker_info(p) for p in persona_list]
    threshold = 0.68

    for idx, utt in enumerate(current_utterances, start=1):
        utt_emb = encode_texts_with_utterance_encoder(
            model=model,
            tokenizer=tokenizer,
            texts=[utt],
            device=device
        )[0]  # shape [H]


        persona_embeds = encode_texts_with_persona_encoder(
            model=model,
            tokenizer=tokenizer,
            texts=filtered_persona_list,
            device=device
        )  # [N, H]

        selected_personas = select_personas_by_threshold(
            utt_emb,
            persona_embeds,
            filtered_persona_list,
            threshold
        )

        print(f"\n[Utterance #{idx}] {utt}")
        print(f" -> Selected Persona(s) (dot >= {threshold}): {selected_personas}")


if __name__ == "__main__":
    main()
