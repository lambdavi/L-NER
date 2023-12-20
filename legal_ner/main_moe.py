"""import torch
import torch.nn as nn
from torch.nn.functional import softmax
from transformers import LukeModel
from transformers.models.luke.configuration_luke import LukeConfig

# Define an Expert Layer
class NERLukeExpert(nn.Module):
    def __init__(self, config):
        super(NERLukeExpert, self).__init__()
        self.luke = LukeModel(config)

    def forward(self, input_ids, attention_mask):
        outputs = self.luke(input_ids=input_ids, attention_mask=attention_mask)
        # Assuming you want to use the last layer's embeddings
        embeddings = outputs.last_hidden_state
        return embeddings

# Define the Dynamic Mixture of Experts Model
class DynamicMoENerLukeModel(nn.Module):
    def __init__(self, config, num_experts, num_labels):
        super(DynamicMoENerLukeModel, self).__init__()
        self.experts = nn.ModuleList([NERLukeExpert(config) for _ in range(num_experts)])
        self.gating_network = nn.Linear(config.hidden_size, num_experts)
        self.final_classifier = nn.Linear(config.hidden_size * num_experts, num_labels)

    def forward(self, input_ids, attention_mask):
        expert_outputs = [expert(input_ids, attention_mask) for expert in self.experts]
        expert_embeddings = torch.cat(expert_outputs, dim=2)

        gating_scores = softmax(self.gating_network(expert_embeddings), dim=-1).unsqueeze(-1)

        # Weighted sum of expert outputs based on gating scores
        weighted_expert_outputs = expert_embeddings * gating_scores
        final_output = torch.sum(weighted_expert_outputs, dim=2)

        # Apply a classifier on the final weighted embeddings
        logits = self.final_classifier(final_output)

        return logits, gating_scores

# Example usage
config = LukeConfig.from_pretrained("studio-ousia/luke-base")
num_experts = 3   # You can adjust the number of experts
original_label_list = [
        "COURT",
        "PETITIONER",
        "RESPONDENT",
        "JUDGE",
        "DATE",
        "ORG",
        "GPE",
        "STATUTE",
        "PROVISION",
        "PRECEDENT",
        "CASE_NUMBER",
        "WITNESS",
        "OTHER_PERSON",
        "LAWYER"
    ]
labels_list = ["B-" + l for l in original_label_list]
labels_list += ["I-" + l for l in original_label_list]
num_labels = len(labels_list) + 1  # Assuming labels_list is defined in your code

dynamic_moe_model = DynamicMoENerLukeModel(config, num_experts, num_labels)

# Assuming you have input_ids and attention_mask as input tensors
input_ids_1 = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])  # Replace with your actual values
attention_mask_1 = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])  # Replace with your actual values

# Sample input_ids and attention_mask for instance 2
input_ids_2 = torch.tensor([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])  # Replace with your actual values
attention_mask_2 = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])  # Replace with your actual values

# Example usage of the model
logits_1, gating_scores_1 = dynamic_moe_model(input_ids_1, attention_mask_1)
logits_2, gating_scores_2 = dynamic_moe_model(input_ids_2, attention_mask_2)

# Printing the results for instance 1
print("Logits for Instance 1:")
print(logits_1)
print("\nGating Scores for Instance 1:")
print(gating_scores_1)
"""
import os
import json
import numpy as np
from argparse import ArgumentParser
from nervaluate import Evaluator
from transformers import LukeModel, LukeTokenizer
from transformers import EarlyStoppingCallback
from transformers import Trainer, TrainingArguments
from torch.nn.functional import softmax
import torch
import torch.nn as nn

from utils.dataset import LegalNERTokenDataset

import spacy
nlp = spacy.load("en_core_web_sm")

class MoEModel(nn.Module):
    def __init__(self, config, num_experts, num_labels):
        super(MoEModel, self).__init__()
        self.luke = LukeModel(config)
        self.experts = nn.ModuleList([nn.Linear(config.hidden_size, num_labels) for _ in range(num_experts)])
        self.gating_network = nn.Linear(config.hidden_size, num_experts)

    def forward(self, input_ids, attention_mask):
        luke_outputs = self.luke(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Gating mechanism
        gating_scores = softmax(self.gating_network(luke_outputs), dim=-1).unsqueeze(-1)

        # Expert outputs
        expert_outputs = [expert(luke_outputs) for expert in self.experts]

        # Weighted sum of expert outputs based on gating scores
        weighted_expert_outputs = torch.cat([score * output for score, output in zip(gating_scores, expert_outputs)], dim=-1)
        final_output = torch.sum(weighted_expert_outputs, dim=-1)

        return final_output, gating_scores

if __name__ == "__main__":

    parser = ArgumentParser(description="Training of MoE model with LUKE base")
    parser.add_argument(
        "--ds_train_path",
        help="Path of train dataset file",
        default="data/NER_TRAIN/NER_TRAIN_ALL.json",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--ds_valid_path",
        help="Path of validation dataset file",
        default="data/NER_DEV/NER_DEV_ALL.json",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--output_folder",
        help="Output folder",
        default="results/",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--batch",
        help="Batch size",
        default=1,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--workers",
        help="Number of workers",
        default=4,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--num_epochs",
        help="Number of training epochs",
        default=5,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--lr",
        help="Learning rate",
        default=1e-5,
        required=False,
        type=float,
    )
    parser.add_argument(
        "--weight_decay",
        help="Weight decay",
        default=0.01,
        required=False,
        type=float,
    )
    parser.add_argument(
        "--warmup_ratio",
        help="Warmup ratio",
        default=0.06,
        required=False,
        type=float,
    )

    args = parser.parse_args()

    # Parameters
    ds_train_path = args.ds_train_path
    ds_valid_path = args.ds_valid_path
    output_folder = args.output_folder
    batch_size = args.batch
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay
    warmup_ratio = args.warmup_ratio
    workers = args.workers

    # Define the labels
    original_label_list = [
        "COURT", "PETITIONER", "RESPONDENT", "JUDGE", "DATE", "ORG", "GPE",
        "STATUTE", "PROVISION", "PRECEDENT", "CASE_NUMBER", "WITNESS", "OTHER_PERSON", "LAWYER"
    ]
    labels_list = ["B-" + l for l in original_label_list]
    labels_list += ["I-" + l for l in original_label_list]
    num_labels = len(labels_list) + 1
    def compute_metrics(pred):
        # Preds
        predictions = np.argmax(pred.predictions, axis=-1)
        predictions = np.concatenate(predictions, axis=0)
        prediction_ids = [[idx_to_labels[p] if p != -100 else "O" for p in predictions]]

        # Labels
        labels = pred.label_ids
        labels = np.concatenate(labels, axis=0)
        labels_ids = [[idx_to_labels[p] if p != -100 else "O" for p in labels]]
        unique_labels = list(set([l.split("-")[-1] for l in list(set(labels_ids[0]))]))
        unique_labels.remove("O")

        # Evaluator
        evaluator = Evaluator(
            labels_ids, prediction_ids, tags=unique_labels, loader="list"
        )
        results, results_per_tag = evaluator.evaluate()

        return {
            "f1-type-match": 2
            * results["ent_type"]["precision"]
            * results["ent_type"]["recall"]
            / (results["ent_type"]["precision"] + results["ent_type"]["recall"] + 1e-9),
            "f1-partial": 2
            * results["partial"]["precision"]
            * results["partial"]["recall"]
            / (results["partial"]["precision"] + results["partial"]["recall"] + 1e-9),
            "f1-strict": 2
            * results["strict"]["precision"]
            * results["strict"]["recall"]
            / (results["strict"]["precision"] + results["strict"]["recall"] + 1e-9),
            "f1-exact": 2
            * results["exact"]["precision"]
            * results["exact"]["recall"]
            / (results["exact"]["precision"] + results["exact"]["recall"] + 1e-9),
        }
    # Load LUKE configuration
    luke_config = LukeModel.from_pretrained("studio-ousia/luke-base").config

    # Create MoE model
    moe_model = MoEModel(luke_config, num_experts=3, num_labels=num_labels)

    # Load datasets
    train_ds = LegalNERTokenDataset(ds_train_path, "studio-ousia/luke-base", labels_list=labels_list, split="train", use_roberta=False)
    val_ds = LegalNERTokenDataset(ds_valid_path, "studio-ousia/luke-base", labels_list=labels_list, split="val", use_roberta=False)

    # Output folder
    new_output_folder = os.path.join(output_folder, 'moe_luke_base')
    if not os.path.exists(new_output_folder):
        os.makedirs(new_output_folder)

    idx_to_labels = {v[1]: v[0] for v in train_ds.labels_to_idx.items()}

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=new_output_folder,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=False,
        fp16_full_eval=False,
        metric_for_best_model="f1-strict",
        dataloader_num_workers=workers,
        dataloader_pin_memory=True,
        report_to="wandb",
        logging_steps=50,
    )

    # Trainer
    trainer = Trainer(
        model=moe_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(5)]
    )

    # Train the model and save it
    trainer.train()
    trainer.save_model(output_folder)
    trainer.evaluate()
