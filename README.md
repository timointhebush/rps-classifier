
# 가위, 바위, 보 이미지 분류하기

가위, 바위, 보 이미지가 input으로 주어졌을 때 딥러닝 모델이 classify하여 이기는 수를 출력하는 application입니다.
- 0: 주먹
- 1: 가위
- 2: 보자기

## Tech Stack

- Python 3.7.12
    - PyTorch 1.10.0
        - torchvision 0.11.2
    - PyTorch Ignite 0.4.9
    - Pandas, numpy
    - PIL
## Train
Clone the project

```bash
  git clone https://github.com/timointhebush/rps-classifier.git
```

Go to the project directory

```bash
  cd rps-classifier
```

Prepare dataset for training
```
.
├── classification
│   ├── dataset
│   │   └── rps
│   │       ├── test
│   │       │   ├── paper
│   │       │   │   ├── 1.png
│   │       │   │   ├── 2.png
│   │       │   │   └── 3.png
│   │       │   ├── rock
│   │       │   │   ├── 1.png
│   │       │   │   ├── 2.png
│   │       │   │   └── 3.png
│   │       │   └── scissors
│   │       │       ├── 1.png
│   │       │       ├── 2.png
│   │       │       └── 3.png
│   │       └── train
│   │           ├── paper
│   │           │   ├── 1.png
│   │           │   ├── 2.png
│   │           │   └── 3.png
│   │           ├── rock
│   │           │   ├── 1.png
│   │           │   ├── 2.png
│   │           │   └── 3.png
│   │           └── scissors
│   │               ├── 1.png
│   │               ├── 2.png
│   │               └── 3.png
```

Train
- 현재 이미지 사이즈 64 * 64 를 기준으로 model layer가 설정되어 있습니다. 

```bash
> python train.py
usage: train.py [-h] --model_fn MODEL_FN [--gpu_id GPU_ID]
                [--train_ratio TRAIN_RATIO] [--valid_ratio VALID_RATIO]
                [--test_ratio TEST_RATIO] [--batch_size BATCH_SIZE]
                [--n_epochs N_EPOCHS] [--verbose VERBOSE]
                [--model_name MODEL_NAME] [--dataset_name DATASET_NAME]
                [--n_classes N_CLASSES] [--freeze] [--use_pretrained]
```
## Predict

- .png 기준
```bash
  python test.py ./{predict 이미지 dir}
```

check `./output.txt`
- `001.png`: 가위, `002.png`: 주먹 일 경우 승리할 수 있는 경우 출력
```tsv
  001	0
  002	1
```

