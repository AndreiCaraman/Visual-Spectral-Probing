import torch
import numpy as np
from evaluate import load
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor

feature_extractor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")


def transform(image_batch):
    # Take a list of PIL images and turn them to pixel values
    image_batch["pixel_values"] = feature_extractor(
        [image.convert("RGB") for image in image_batch["image"]], return_tensors="pt"
    )["pixel_values"]
    del image_batch["image"]
    return image_batch


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }


metric = load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


# utility function for randomly displaying dataset images
def display_images(ds, title):
    fig, axs = plt.subplots(4, 4, figsize=(20, 10))
    axs = axs.flatten()

    ds = ds.shuffle().select(range(16))

    for idx, ax in enumerate(axs):
        ax.imshow(np.array(ds[idx]["image"]))

        label_id = ds[idx]["label"]
        ax.set_title(ds.features["label"].int2str(label_id))

        ax.axis("off")

    fig.suptitle(title)
    plt.show()
