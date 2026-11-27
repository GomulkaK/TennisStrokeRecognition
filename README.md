# Comparative Analysis of Deep Learning Models for Subject-Independent Tennis Stroke Recognition

This repository contains the source code, data split configurations, class labels, and selected model checkpoints accompanying the paper **"Comparative Analysis of Deep Learning Models for Subject-Independent Tennis Stroke Recognition from RGB Videos"**.

The study investigates the performance of different deep learning architectures for recognizing twelve tennis stroke categories from RGB videos of the THETIS dataset. Particular emphasis is placed on subject-independent evaluation, in which players appearing in the test set are not present in the training set.

Nine model configurations were evaluated, covering three main approaches to spatiotemporal feature extraction:

- **Frame-based CNN–LSTM models:** InceptionV3–LSTM and Xception–LSTM.
- **Spatiotemporal convolutional networks:** R(2+1)D-18.
- **Transformer-based video architectures:** Swin3D-Tiny, Swin3D-Base, MViT-v1-B, MViT-v2-S, TimeSformer, and TimeSformer-HR.

The repository provides the experimental configurations and supporting code used in the study to facilitate reproducibility and further research on video-based tennis stroke recognition.

## Dataset

The experiments were conducted using the RGB videos from the publicly available [THETIS (THree dimEnsional TennIs Shots) dataset](LINK_TO_THETIS_DATASET).

The dataset contains 1,980 RGB video clips representing twelve tennis stroke categories, performed by 55 subjects (31 amateur and 24 experienced players).

The following twelve classes were considered:

| Class ID | Abbreviation | Class name |
|---:|:---:|---|
| 0 | BH | Backhand |
| 1 | BH2 | Backhand with two hands |
| 2 | BHS | Backhand slice |
| 3 | BHV | Backhand volley |
| 4 | SF | Service flat |
| 5 | FF | Forehand flat |
| 6 | FOS | Forehand open stands |
| 7 | FS | Forehand slice |
| 8 | FV | Forehand volley |
| 9 | SK | Service kick |
| 10 | SS | Service slice |
| 11 | SM | Smash |

**Note:** The original THETIS video files are not included in this repository. Please obtain the dataset from its original source and organize the RGB videos according to the paths specified in the provided split files.

## Data Splits and Class Labels

The repository includes the data split configurations used in the experiments. Three evaluation settings are provided:

- **Subject-independent single split (70/20/10):** Training, validation, and test sets contain disjoint groups of subjects.
- **Subject-independent five-fold cross-validation:** Five subject-independent folds are provided to evaluate model performance across different subject partitions.
- **Subject-dependent single split (70/20/10):** Subjects may occur in multiple subsets, allowing the effect of subject overlap on classification performance to be investigated.

The split files specify the video paths and corresponding class IDs. The same data partitions were used across the evaluated architectures to ensure a consistent comparison.

### Directory Structure

The data split configurations and class labels are organized as follows:

```text
├── splits/
│   ├── subject_independent/
│   │   ├── single_split/
│   │   │   ├── train.csv
│   │   │   ├── val.csv
│   │   │   └── test.csv
│   │   │
│   │   └── five_fold/
│   │       ├── fold_1/
│   │       │   ├── train.csv
│   │       │   ├── val.csv
│   │       │   └── test.csv
│   │       ├── fold_2/
│   │       ├── fold_3/
│   │       ├── fold_4/
│   │       └── fold_5/
│   │
│   └── subject_dependent/
│       └── single_split/
│           ├── train.csv
│           ├── val.csv
│           └── test.csv
│
├── labels/
│   └── class_labels.csv
```

The CSV split files use the following format:

```text
video_path|class_id
```

Each row represents one video sample, with its corresponding path and numerical class label. The files do not contain a header row.

The paths stored in the split files should be adjusted to match the local location of the THETIS RGB video directory.

## Source Code

The repository contains code for the evaluated architectures and the main stages of the experimental workflow.

### 1. CNN–LSTM Models

The InceptionV3–LSTM and Xception–LSTM architectures use a two-stage processing procedure:

1. **Feature extraction:** Individual RGB frames are processed using an ImageNet-pretrained CNN backbone. The resulting frame-level feature vectors are arranged into fixed-length temporal sequences and saved for subsequent training.
2. **LSTM training:** The previously extracted feature sequences are used to train the recurrent classification networks.

This separation allows feature extraction to be performed independently of recurrent model training.

### 2. Spatiotemporal Video Models

The repository includes training and evaluation code for the following video-based architectures:

- R(2+1)D-18
- Swin3D-Tiny
- Swin3D-Base
- MViT-v1-B
- MViT-v2-S

These models process video clips directly and jointly learn spatial and temporal representations.

### 3. TimeSformer Models

The TimeSformer and TimeSformer-HR models were trained using the [official TimeSformer implementation](https://github.com/facebookresearch/TimeSformer).

The repository includes the corresponding training configurations and supporting modifications used in the experiments.

## Pretrained Checkpoints

The models were initialized using publicly available pretrained weights, where applicable.

The repository may include selected fine-tuned checkpoints to provide an example of the trained models. Checkpoints for all evaluated architectures are not necessarily distributed.

For the TimeSformer models, the original Kinetics-400 pretrained checkpoints can be obtained from the [official TimeSformer repository](https://github.com/facebookresearch/TimeSformer).

## Experimental Evaluation

The models were evaluated using the same data partitions and evaluation metrics to ensure a fair comparison.

The study reports results for the following protocols:

- Subject-independent single train-validation-test split.
- Subject-independent five-fold cross-validation.
- Subject-dependent train-validation-test split.

The evaluation includes overall classification performance, per-class results, and model complexity-related measures.

## Citation

If you use the code, data split configurations, or other resources provided in this repository, please cite the accompanying paper:

```bibtex
@article{krzeszowski2026tennis,
  title={Comparative Analysis of Deep Learning Models for Subject-Independent Tennis Stroke Recognition from RGB Videos},
  author={Krzeszowski, Tomasz and Gomulka, Kamil and Powroznik, Pawel and Skublewska-Paszkowska, Maria},
  journal={Applied Sciences},
  year={2026}
}
```

*The final bibliographic information and DOI will be added after publication.*

## Acknowledgements

This work uses the THETIS dataset and builds upon several publicly available deep learning architectures and implementations, including TimeSformer.

We thank the original dataset authors and the developers of the referenced architectures for making their resources available to the research community.

## License

The source code in this repository is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.

The THETIS dataset, pretrained model weights, and third-party implementations remain subject to their respective licenses and terms of use.
