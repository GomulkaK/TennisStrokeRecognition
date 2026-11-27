# Comparative Analysis of Deep Learning Models for Subject-Independent Tennis Stroke Recognition

This repository contains the source code, data split configurations, class labels, and selected model checkpoints accompanying the paper **"Comparative Analysis of Deep Learning Models for Subject-Independent Tennis Stroke Recognition from RGB Videos"**.

The study investigates the performance of different deep learning architectures for recognizing twelve tennis stroke categories from RGB videos of the THETIS dataset. Particular emphasis is placed on subject-independent evaluation, in which players appearing in the test set are not present in the training set.

Nine model configurations were evaluated, covering three main approaches to spatiotemporal feature extraction:

- **Frame-based CNN–LSTM models:** InceptionV3–LSTM and Xception–LSTM.
- **Spatiotemporal convolutional networks:** R(2+1)D-18.
- **Transformer-based video architectures:** Swin3D-Tiny, Swin3D-Base, MViT-v1-B, MViT-v2-S, TimeSformer, and TimeSformer-HR.

The repository provides the experimental configurations and supporting code used in the study to facilitate reproducibility and further research on video-based tennis stroke recognition.

## Dataset

The experiments were conducted using the RGB videos from the publicly available [THETIS (THree dimEnsional TennIs Shots) dataset](https://github.com/THETIS-dataset/dataset).
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

After downloading the THETIS dataset from its [original repository](https://github.com/THETIS-dataset/dataset), extract the RGB video files into the `VIDEO_RGB/` directory in the root of this repository. Preserve the original class subdirectory names so that they match the paths specified in the provided split files.

The expected directory structure is:

```text
repository/
├── VIDEO_RGB/
│   ├── backhand/
│   ├── backhand2hands/
│   └── ...
├── labels/
├── splits/
└── torchvision_video.py
```

If the videos are stored in a different location, specify the appropriate directory using the `--data-dir` argument when running the training or evaluation script.

## Installation and Requirements

The experiments were implemented in Python using PyTorch and related computer vision and machine learning libraries. The dependencies required to run the scripts include:

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Pillow
- OpenCV
- scikit-learn
- Matplotlib
- tqdm

The exact PyTorch and Torchvision versions should be selected according to the installed Python version and hardware configuration. For GPU acceleration, install a PyTorch build compatible with the available CUDA version. Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for the appropriate installation command.

### Basic environment setup

Create and activate a virtual environment:

```bash
python -m venv tennis_env
```

On Windows:

```bash
tennis_env\Scripts\activate
```

On Linux:

```bash
source tennis_env/bin/activate
```

Install PyTorch and Torchvision using the command recommended by the official PyTorch installation guide. Then install the remaining common dependencies:

```bash
pip install numpy pandas pillow opencv-python scikit-learn matplotlib tqdm
```

### Additional dependencies for TimeSformer

The TimeSformer models use the [official TimeSformer implementation](https://github.com/facebookresearch/TimeSformer), which requires additional packages:

```bash
pip install fvcore simplejson einops timm av psutil tensorboard
pip install scikit-learn opencv-python
pip install "git+https://github.com/facebookresearch/fvcore"
```

The official implementation also requires building and installing the TimeSformer codebase:

```bash
git clone https://github.com/facebookresearch/TimeSformer
cd TimeSformer
python setup.py build develop
```

Refer to the original repository for further installation details.

**Note:** The TimeSformer implementation is an older, archived codebase. Its dependencies may require a separate Python environment compatible with the original implementation. The other models in this repository use the standard PyTorch and Torchvision libraries.
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

#### Training and Evaluation

The `torchvision_video.py` script provides a unified interface for training and evaluating the five torchvision video architectures: R(2+1)D-18, Swin3D-Tiny, Swin3D-Base, MViT-v1-B, and MViT-v2-S.

The script supports both training and evaluation using the provided data splits. The model architecture, evaluation protocol, split configuration, and fold can be selected through command-line arguments.

**Example – training MViT-v2-S using the subject-independent single split:**

```bash
python train_torchvision_video.py --mode train --model mvit_v2_s --protocol subject_independent --split-mode single_split
```

**Example – training Swin3D-Tiny using the third fold of subject-independent five-fold cross-validation:**

```bash
python train_torchvision_video.py --mode train --model swin3d_t --protocol subject_independent --split-mode five_fold --fold 3
```

**Example – evaluating a previously trained MViT-v2-S checkpoint:**

```bash
python train_torchvision_video.py --mode test --model mvit_v2_s --protocol subject_independent --split-mode single_split --checkpoint "outputs/mvit_v2_s/subject_independent/single_split/mvitv2s_best.pth"
```

In evaluation mode, the script loads the specified checkpoint and evaluates it on the corresponding test split without additional training. Training and evaluation results, including checkpoints, training history, classification reports, and confusion matrices, are saved in the `outputs/` directory.

The default video directory is `VIDEO_RGB/` in the repository root. An alternative location can be specified using the `--data-dir` argument.

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
