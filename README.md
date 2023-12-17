# Visual Spectral Probing

This archive contains implementations of the methods from the masters's thesis "**Visual Spectral Probing**".

It enables probing for the Vision Transformer using unfiltered, manually filtered and automatically filtered representations using the following filter definitions:

The toolkit is relatively flexible and can be applied to any model size or dataset availiable at the Hugging Face transformers library.

After installing the required packages, and downloading external datasets, the experiments can be re-run using the `train.py` script. Please see the instructions below for details.

## Installation

This repository uses Python 3.11.6 and the associated packages listed in the `requirements.txt` (a virtual environment is recommended):

```bash
(venv) $ pip install -r requirements.txt
```

## Data

TODO:
- COCO Backgrounds
- food101

## Experiments

Running an experiment involves training a classification head together with a specified filter (see `train.py` for details):

```bash
(venv) $ python train.py
```

### COCO Backgrounds

TODO:
The following lists the experiments included in this repository:

* **filter 33 threshold** located in `tasks/20news/` requires a spearate download of the original data (please use the version in `20news-bydate.tar.gz`).
