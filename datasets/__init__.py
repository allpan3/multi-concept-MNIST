from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from . import MultiConceptMNIST

def collate_train_fn(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    labels = [x[1] for x in batch]
    targets = torch.stack([x[2] for x in batch], dim=0)
    return imgs, labels, targets

def collate_test_fn(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    labels = [x[1] for x in batch]
    targets = torch.stack([x[2] for x in batch], dim=0)
    questions = [x[3] for x in batch]
    return imgs, labels, targets, questions

def get_train_data(root, vsa = None, num_samples = 10000, max_num_objects = 3, single_count = False, batch_size = 64, num_pos_x = 3, num_pos_y = 3, num_colors = 7, transform = None):
    train_ds = MultiConceptMNIST(root, vsa, train=True, num_samples=num_samples, max_num_objects=max_num_objects, single_count=single_count, num_pos_x=num_pos_x, num_pos_y=num_pos_y, num_colors=num_colors, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_train_fn)
    return train_dl

def get_test_data(root, vsa = None, num_samples = 100, max_num_objects = 3, single_count = False, batch_size = 1, num_pos_x = 3, num_pos_y = 3, num_colors = 7, transform = None):
    test_ds = MultiConceptMNIST(root, vsa, train=False, num_samples=num_samples, max_num_objects=max_num_objects, single_count=single_count, num_pos_x=num_pos_x, num_pos_y=num_pos_y, num_colors=num_colors, transform=transform)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_test_fn)
    return test_dl