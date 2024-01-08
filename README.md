# Visual Spectral Probing

This archive contains implementations of the methods from the masters's thesis "**Visual Spectral Probing**".

It enables probing for the Vision Transformer using unfiltered, manually filtered and automatically filtered representations using the following filter definitions:

1. **Filter Bypass Mode:**
   No filtering is performed, allowing an assessment of the original contextual representations.

2. **Auto-Filter Mode:**
   The filter dynamically adjusts its weights based on the existing training objective, jointly learning which frequencies to amplify or filter out. This mode aims to optimize the model's performance on the classification task.

3. **Band-Filter Mode:**
   Indicated frequency bands are selectively removed from the contextual representations. This mode provides insights into how specific frequency ranges contribute to the overall understanding of the image.

The toolkit is relatively flexible and can be applied to any model size or dataset availiable at the Hugging Face transformers library.

After installing the required packages, and downloading external datasets, the experiments can be re-run using the `train.py` script. Please see the instructions below for details.

## Installation

This repository uses Python 3.11.6 and the associated packages listed in the `requirements.txt` (a virtual environment is recommended):

```bash
(venv) $ pip install -r requirements.txt
```

## Data

### COCO Backgrounds

To explore the role of low-frequency components in contextual representations for image classification, a specialized dataset, COCO Backgrounds, is curated. Derived from the COCO-Stuff dataset, it augments COCO 2017 images with pixel-wise annotations for 91 stuff classes. The dataset preparation involves a three-step approach, emphasizing meaningful labeling and resulting in a dataset tailored for studying low-frequency components. Various experiments, employing different thresholds, demonstrate the dataset's suitability for probing experiments.

For image examples, refer to the relative notebook.

### Food 101

To complement the low-frequency analysis, the Food 101 dataset is employed. With 101 food categories and 101,000 images, it aligns with the goal of understanding scale-specific information in image embeddings. Selected for structured patterns and minimal background variability, the dataset aims to evaluate how disentangling high-frequency components influences the model's classification of food items. The focus is on uncovering insights into the impact of scale-specific information on classifying items with distinct visual structures.

For image examples, refer to relative notebook.


## Experiments

Running an experiment involves training a classification head together with a specified filter (see `train.py` for details):

```bash
(venv) $ python train.py
```

## Results

Experiment runs for differnt configurations can be viewed at the following links:

- [ViTMAEBase-Probing-COCObackground](https://wandb.ai/team-mirko/ViTMAEBase-Probing-COCObackground)
- [ViTMAEHuge-Probing-COCObackground](https://wandb.ai/team-mirko/ViTMAEHuge-Probing-COCObackground)
- [ViTMAEBase-Probing-Food101](https://wandb.ai/team-mirko/ViTMAEBase-Probing-Food101)
- [ViTMAEHuge-Probing-Food101](https://wandb.ai/team-mirko/ViTMAEHuge-Probing-Food101)
