#!/usr/bin/env python
# coding: utf-8

project_name = "ViTMAEBase-Probing-COCObackground"
# run_name = "auto-filter-50threshold"
run_name = "no-filter"

import os, sys, wandb

sys.path.append("./src")
os.environ["WANDB_SILENT"] = "true"

from datasets import load_dataset
from utils import transform
from models import PrismViTConfig, PrismViT
from transformers import Trainer, TrainingArguments
from utils import collate_fn, compute_metrics
from torch.optim import SGD
from callback import WandbCustomCallback

# loading data and applying preprocessing to have suitable input format
ds_train, ds_val = load_dataset(
    "./coco_backgrounds",
    # "food101",
    split=["train", "validation"],
    keep_in_memory=True,
)
preprocessed_train_ds = ds_train.with_transform(transform)
preprocessed_val_ds = ds_val.with_transform(transform)
labels = ds_train.features["label"].names

wandb.init(
    project=project_name,
    name=run_name,
    # resume="must",
    # id="id",
)

configuration = PrismViTConfig(
    # filter_type = {'filter_name': 'auto', 'filter_args': [257]}, # for ViTHuge
    # filter_type={"filter_name": "auto", "filter_args": [197]},  # for ViTBase
    # filter_type={"filter_name": "eqalloc", "filter_args": [197, bands, bi]},
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
)

model = PrismViT(configuration)

# freezing the backbone
model.train()
for param in model.base_model.parameters():
    param.requires_grad = False

# using optimizer suggestet by ViTMAE paper
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

args = TrainingArguments(
    "./artifacts/" + str(project_name + "-" + run_name),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    dataloader_num_workers=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    remove_unused_columns=False,
    save_total_limit=2,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model,
    args=args,
    data_collator=collate_fn,
    train_dataset=preprocessed_train_ds,
    eval_dataset=preprocessed_val_ds,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
)

# custom callback to plot filter weights in wandb
# trainer.add_callback(WandbCustomCallback(trainer=trainer))

trainer_output = trainer.train(
    # resume_from_checkpoint=True
)

wandb.finish()
