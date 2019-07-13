from PIL import Image
import torch
from torch.utils.data import Dataset


class BBoxDataset(Dataset):
    """PyTorch datasets for object detection.

    Parameters
    ----------
    images : list of str
        A List of image's paths.

    bboxes: list of torch.Tensor
        Each tensor's shape: [[x_min, y_min, x_max, y_max], [x_min, ...], ...]

    labels: list of torch.Tensor
        Each tensor's shape: [] or [0, 3] or [0, 4 ,5] ...

    transform: torchvision.transforms
    """

    def __init__(self, images, bboxes, labels, transform):
        self.images = images
        self.bboxes = bboxes
        self.labels = labels
        self.transform = transform

    def _path_to_tensor(self, path):
        img = Image.open(path).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self._path_to_tensor(self.images[idx])
        bbox = self.bboxes[idx]
        label = self.labels[idx]
        target = {'boxes': bbox.float(), 'labels': label.int()}
        return img, target


def bbox_collate_fn(batch):
    images = []
    targets = []
    for sample in batch:
        image, target = sample
        images.append(image)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets