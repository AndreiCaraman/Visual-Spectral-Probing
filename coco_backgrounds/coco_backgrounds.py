# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""COCOBackgrounds"""
import json
import os
from pathlib import Path

import datasets

_THRESHOLD = 0.50

_CITATION = """
@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and
               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{\'{a}}r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
  url       = {http://arxiv.org/abs/1405.0312},
  eprinttype = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/LinMBHPRDZ14.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """
MS COCO is a large-scale object detection, segmentation, and captioning dataset.
COCO has several features: Object segmentation, Recognition in context, Superpixel stuff segmentation, 330K images (>200K labeled), 1.5 million object instances, 80 object categories, 91 stuff categories, 5 captions per image, 250,000 people with keypoints.
"""

_HOMEPAGE = "https://cocodataset.org/#home"

_LICENSE = "CC BY 4.0"


_IMAGES_URLS = {
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "validation": "http://images.cocodataset.org/zips/val2017.zip",
}

_ANOTATION_FILE_URL = (
    "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
)

COCOSTUFF_SUPERCAATEGORY_CLASSES = [
    "plant",
    "textile",
    "floor",
    "ground",
    "furniture-stuff",
    "food-stuff",
    "structural",
    "window",
    "building",
    "wall",
    "solid",
    "ceiling",
    "sky",
    "raw-material",
    "water",
]

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "label": datasets.ClassLabel(names=COCOSTUFF_SUPERCAATEGORY_CLASSES),
        # "image_id": datasets.Value("int64"),
        # "height": datasets.Value("int64"),
        # "width": datasets.Value("int64"),
        # "file_name": datasets.Value("string"),
        # "coco_url": datasets.Value("string"),
        # "image_path": datasets.Value("string"),
    }
)

class CocoBackgrounds(datasets.GeneratorBasedBuilder):
    """Subset of COCO with image labels the backgrounds"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        annotation_files_dir_path = os.path.join(
            dl_manager.download_and_extract(_ANOTATION_FILE_URL), "annotations"
        )
        image_folders = {
            k: Path(v) for k, v in dl_manager.download_and_extract(_IMAGES_URLS).items()
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotation_file": os.path.join(
                        annotation_files_dir_path, "panoptic_train2017.json"
                    ),
                    "image_folder": os.path.join(image_folders["train"], "train2017"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "annotation_file": os.path.join(
                        annotation_files_dir_path, "panoptic_val2017.json"
                    ),
                    "image_folder": os.path.join(
                        image_folders["validation"], "val2017"
                    ),
                },
            ),
        ]

    def _generate_examples(self, annotation_file, image_folder):
        counter = 0

        with open(annotation_file, "r", encoding="utf-8") as fi:
            ds_metadata = json.load(fi)

        patch_id2label = self.build_patch_id2label(ds_metadata)
        image_id2file_name = {
            image["id"]: image["file_name"] for image in ds_metadata["images"]
        }

        for image in ds_metadata["annotations"]:
            patches = image["segments_info"]

            stuffPatches = [
                patch
                for patch in patches
                if patch["category_id"] in patch_id2label.keys()
            ]
            image_area = sum([patch["area"] for patch in patches])

            # keeping in the dataset only the images that have a stuff object as a single non ambigous subject
            # i.e. the normalized patch area si greater that a fixed threshold
            relevantPatches = [
                patch
                for patch in stuffPatches
                if (patch["area"] / image_area) > _THRESHOLD
            ]

            if len(relevantPatches) == 1:
                image_file_name = image_id2file_name[image["image_id"]]
                image_path = os.path.join(image_folder, image_file_name)

                label = patch_id2label[relevantPatches.pop()["category_id"]]

                yield counter, {
                    "image": image_path,
                    "label": label,
                }
                counter += 1

    def build_patch_id2label(self, data):
        patch_id2label = dict()

        for category in data["categories"]:
            if not category["isthing"]:
                patch_id = category["id"]
                label = category["supercategory"]

                patch_id2label[patch_id] = label

        return patch_id2label
