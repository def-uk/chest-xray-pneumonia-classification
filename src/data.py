"""Dataset paths, a reproducible stratified split, and image loaders."""
# 흉부 X-ray(폐렴 분류) 데이터셋의 경로 처리, 재현 가능한 층화(stratified) 분할,
# 그리고 train/val/test DataLoader 생성을 담당하는 모듈.
from pathlib import Path
import re

from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2


def dataset_dirs(root):
    """데이터셋 루트에서 train/val/test 폴더 경로를 딕셔너리로 반환한다."""
    root = Path(root)
    # 압축을 풀면 'chest_xray/' 폴더가 한 단계 더 생기는 경우가 있으므로,
    # 그 폴더가 존재하면 실제 루트를 그 안쪽으로 옮긴다.
    candidate = root / 'chest_xray'
    if candidate.is_dir():
        root = candidate
    # {'train': .../train, 'val': .../val, 'test': .../test}
    return {name: root / name for name in ('train', 'val', 'test')}


def make_transforms(augment=False, pretrained=False):
    """이미지 전처리 파이프라인을 만든다.

    augment:    True면 학습용 데이터 증강(작은 회전)을 추가한다.
    pretrained: True면 ImageNet 사전학습 모델용 정규화 값을 사용한다.
    """
    # X-rays are grayscale; pretrained ImageNet networks require three channels.
    # 사전학습 모델이면 ImageNet 통계값, 아니면 [-1, 1] 범위로 맞추는 0.5/0.5 사용.
    mean = (0.485, 0.456, 0.406) if pretrained else (0.5, 0.5, 0.5)
    std = (0.229, 0.224, 0.225) if pretrained else (0.5, 0.5, 0.5)

    # 1) 흑백 이미지를 3채널로 복제 (ImageNet 모델 입력 형식에 맞춤)
    # 2) 모든 이미지를 224x224로 크기 통일
    steps = [v2.Grayscale(num_output_channels=3), v2.Resize((224, 224))]

    # 학습용일 때만 ±5도 랜덤 회전 증강
    # (X-ray는 촬영 방향이 거의 일정하므로 증강을 약하게 둔다)
    if augment:
        steps.append(v2.RandomRotation(5))

    # 3) PIL 이미지 → 텐서 변환
    # 4) float32로 바꾸면서 픽셀값을 [0, 255] → [0, 1]로 스케일링
    # 5) 채널별 평균/표준편차로 정규화
    steps += [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    return v2.Compose(steps)


def patient_group(image_path, label):
    """Extract the patient identifier encoded in this dataset's filenames."""
    stem = Path(image_path).stem
    if label == 'PNEUMONIA':
        match = re.match(r'person\d+', stem, re.IGNORECASE)
    elif label == 'NORMAL':
        match = re.match(r'(?:NORMAL2-)?IM-\d+', stem, re.IGNORECASE)
    else:
        raise ValueError(f'Unexpected class: {label}')
    if match is None:
        raise ValueError(f'Cannot identify patient from filename: {image_path}')
    return f'{label}:{match.group().upper()}'


def make_loaders(root, batch_size=32, val_fraction=0.2, seed=42,
                 augment=False, pretrained=False, num_workers=0):
    """Split original train+val; keep the provided test set untouched."""
    # 원본 데이터셋의 val 폴더는 이미지가 너무 적어서(클래스당 몇 장)
    # train + val을 합친 뒤 클래스 비율을 유지하며 다시 나눈다.
    # test 폴더는 최종 평가용이므로 전혀 건드리지 않는다.
    dirs = dataset_dirs(root)

    # 라벨 정보만 얻기 위해 transform 없이 먼저 읽는다.
    plain = datasets.ImageFolder(dirs['train'])
    original_val = datasets.ImageFolder(dirs['val'])

    # ImageFolder는 폴더 이름 알파벳순으로 클래스 번호를 매기므로,
    # 두 폴더의 클래스 구성이 다르면 라벨 번호가 어긋난다 → 미리 검사.
    if plain.classes != original_val.classes:
        raise ValueError('Train and validation class names differ')

    # train 라벨 뒤에 val 라벨을 이어붙임.
    # 이 순서는 아래 ConcatDataset([train, val])의 인덱스 순서와 정확히 일치해야 한다.
    labels = plain.targets + original_val.targets

    # 한 환자의 여러 X-ray가 train과 val 양쪽에 들어가지 않도록 그룹 분할.
    samples = plain.samples + original_val.samples
    groups = [patient_group(path, plain.classes[label]) for path, label in samples]
    n_splits = round(1 / val_fraction)
    if n_splits < 2 or abs(1 / n_splits - val_fraction) > 1e-9:
        raise ValueError('val_fraction must be the reciprocal of an integer >= 2')
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    overall_positive = sum(labels) / len(labels)
    candidates = list(splitter.split(samples, labels, groups))
    # Group sizes vary, so choose the fold closest to the requested fraction
    # and class balance. Every candidate still has disjoint patient groups.
    train_ids, val_ids = min(candidates, key=lambda pair: (
        abs(len(pair[1]) / len(labels) - val_fraction) +
        abs(sum(labels[i] for i in pair[1]) / len(pair[1]) - overall_positive)))
    train_ids, val_ids = train_ids.tolist(), val_ids.tolist()
    if {groups[i] for i in train_ids} & {groups[i] for i in val_ids}:
        raise RuntimeError('Patient overlap between train and validation')

    # 학습용(증강 가능)과 평가용(증강 없음) 전처리를 따로 준비
    train_transform = make_transforms(augment, pretrained)
    eval_transform = make_transforms(False, pretrained)

    # 같은 이미지 풀(train+val)을 transform만 다르게 두 벌 만든다.
    # → train 샘플은 증강이 적용되고, val 샘플은 증강 없이 평가된다.
    train_sources = ConcatDataset([datasets.ImageFolder(dirs['train'], transform=train_transform),
                                   datasets.ImageFolder(dirs['val'], transform=train_transform)])
    eval_sources = ConcatDataset([datasets.ImageFolder(dirs['train'], transform=eval_transform),
                                  datasets.ImageFolder(dirs['val'], transform=eval_transform)])

    # 앞에서 구한 인덱스로 각 풀에서 필요한 샘플만 골라낸다.
    train_set = Subset(train_sources, train_ids)
    val_set = Subset(eval_sources, val_ids)

    # test는 원본 그대로, 평가용 전처리만 적용
    test_set = datasets.ImageFolder(dirs['test'], transform=eval_transform)
    if test_set.classes != plain.classes:
        raise ValueError('Test class names differ')

    return {
        # 학습 데이터만 매 epoch마다 순서를 섞는다.
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        # 검증/테스트는 순서 고정 → 결과 비교와 재현이 쉬움
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        # 클래스 이름 목록 (예: ['NORMAL', 'PNEUMONIA'])
        'classes': plain.classes,
        # 각 분할의 샘플 수 (로그 출력·확인용)
        'sizes': {'train': len(train_set), 'val': len(val_set), 'test': len(test_set)},
        'split': {'strategy': 'stratified_patient_group', 'seed': seed,
                  'train_patients': len({groups[i] for i in train_ids}),
                  'val_patients': len({groups[i] for i in val_ids}),
                  'patient_overlap': 0},
    }
