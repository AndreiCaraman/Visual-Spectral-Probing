from transformers import Trainer, TrainingArguments
from functions import model_init, collate_fn, compute_metrics
from functions import preprocessed_train_ds, preprocessed_val_ds

from time import time
import multiprocessing as mp

for num_workers in range(2, mp.cpu_count(), 2):

    args = TrainingArguments(
        "./temp",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
        report_to='none',
    )

    trainer = Trainer(
        model_init=model_init,
        args=args,
        data_collator = collate_fn,
        train_dataset = preprocessed_train_ds.select(range(10)), # dummy epochs
        eval_dataset = preprocessed_val_ds.select(range(1)), # dummy epochs
        compute_metrics = compute_metrics,
    )

    start = time()

    trainer_output = trainer.train()
    
    end = time()

    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))