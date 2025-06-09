import os.path as osp
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import glob
import re
import os

def _process_dir(dir_path, relabel=False):
    img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')

    pid_container = set()
    for img_path in img_paths:
        pid, _ = map(int, pattern.search(osp.basename(img_path)).groups())
        if pid == -1: continue  # junk images are just ignored
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    dataset = []
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(osp.basename(img_path)).groups())
        if pid == -1: continue  # junk images are just ignored
        assert 0 <= pid <= 1501  # pid == 0 means background
        assert 1 <= camid <= 6
        camid -= 1  # index starts from 0
        if relabel: pid = pid2label[pid]
        dataset.append((img_path, pid, camid))

    return dataset

class Market1501(Dataset):
    def __init__(self, root='', mode='train', transform=None):
        super(Market1501, self).__init__()
        self.transform = transform
        self.loader = default_loader
        self.root = root

        if mode == 'train':
            self.data = _process_dir(osp.join(root, 'bounding_box_train'), relabel=True)
        elif mode == 'query':
            self.data = _process_dir(osp.join(root, 'query'), relabel=False)
        elif mode == 'gallery':
            self.data = _process_dir(osp.join(root, 'bounding_box_test'), relabel=False)
        else:
            raise ValueError('Invalid mode. Got {}, but expected one of [train | query | gallery]'.format(mode))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, pid, camid = self.data[index]
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, img_path 