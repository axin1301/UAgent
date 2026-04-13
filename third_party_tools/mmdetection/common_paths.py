import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY_ROOT = CURRENT_DIR
WEIGHTS_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'weights'))
SUPPORT_ROOT = os.path.join(CURRENT_DIR, 'support')
MMDET_CONFIG_ROOT = os.path.join(SUPPORT_ROOT, 'configs', 'mmdetection')
MMSEG_CONFIG_ROOT = os.path.join(SUPPORT_ROOT, 'configs', 'mmsegmentation')
BACKBONE_SUPPORT_ROOT = os.path.join(SUPPORT_ROOT, 'backbones')
DATASET_SUPPORT_ROOT = os.path.join(SUPPORT_ROOT, 'datasets')


def weight_path(filename: str) -> str:
    return os.path.join(WEIGHTS_ROOT, filename)


def mmdet_config(filename: str) -> str:
    return os.path.join(MMDET_CONFIG_ROOT, filename)


def mmseg_config(filename: str) -> str:
    return os.path.join(MMSEG_CONFIG_ROOT, filename)
