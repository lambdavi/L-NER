import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer, RobertaTokenizerFast
from utils.utils import match_labels

import spacy
nlp = spacy.load("en_core_web_sm")

class LegalNERTokenDataset(Dataset):
    
    def __init__(self, dataset_path, model_path, labels_list=None, split="train", use_roberta=False):
        self.model_path = model_path
        self.data = json.load(open(dataset_path))
        self.split = split
        self.use_roberta = use_roberta
        self.rob_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.ber_tokenizer = AutoTokenizer.from_pretrained(model_path) 
        self.labels_list = sorted(labels_list + ["O"])[::-1]

        if self.labels_list is not None:
            self.labels_to_idx = dict(
                zip(sorted(self.labels_list)[::-1], range(len(self.labels_list)))
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["data"]["text"]
        annotations = [
            {
                "start": v["value"]["start"],
                "end": v["value"]["end"],
                "labels": v["value"]["labels"][0],
            }
            for v in item["annotations"][0]["result"]
        ]

        if self.use_roberta:
            inputs = self.rob_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding='max_length'
            )
        else:
            inputs = self.ber_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding='max_length'
            )

        aligned_labels = match_labels(inputs, annotations)
        aligned_labels = [self.labels_to_idx[l] for l in aligned_labels]

        # Extract tensors from the tokenizer output
        input_ids = inputs["input_ids"].squeeze(0).long()
        attention_mask = inputs["attention_mask"].squeeze(0).long()

        if not self.use_roberta:
            token_type_ids = inputs["token_type_ids"].squeeze(0).long()
        else:
            token_type_ids = None

        if self.labels_list:
            labels = torch.tensor(aligned_labels).squeeze(-1).long()
            if labels.shape[0] < attention_mask.shape[0]:
                pad_x = torch.zeros((attention_mask.shape[0],))
                pad_x[: labels.size(0)] = labels
                target = aligned_labels
            else:
                target = labels[: attention_mask.shape[0]]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": target if self.labels_list else None
        }