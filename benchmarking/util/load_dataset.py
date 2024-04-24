import sh
import pandas as pd
from os.path import isdir, isfile, dirname

SSD_PATH = '/media/anand/Data'


def _mount_(dest=SSD_PATH):
    ssd_mount_origin = '/dev/sda2'
    sh.contrib.sudo.mkdir(dest)
    sh.contrib.sudo.mount(ssd_mount_origin, dest)


def load_dataset(dataset_path):
    if not isdir(dirname(dataset_path)):
        _mount_(SSD_PATH)
    if not isfile(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found in SSD_PATH={SSD_PATH}")
    
    return pd.read_csv(dataset_path)