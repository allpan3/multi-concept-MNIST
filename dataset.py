""" Multi-Concept MNIST Dataset
"""

# %%
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets.vision import VisionDataset
import torch.utils.data as data
import torch
import random
import json
import os
from model.vsa import MultiConceptMNISTVSA
from typing import Any, Callable, Dict, List, Optional, Tuple


class MultiConceptMNIST(VisionDataset):

    COLOR_SET = [
        range(0,3),   # white
        range(0,1),   # red
        range(1,2),   # green
        range(2,3),   # blue
        range(0,2),   # yellow
        range(0,3,2), # magenta
        range(1,3)    # cyan
    ]

    data = []
    labels = []
    targets = []

    def __init__(
        self,
        root: str,
        vsa: MultiConceptMNISTVSA,
        train: bool,      # training set or test set
        num_samples: int,
        force_gen: bool = False,  # generate dataset even if it exists
        max_num_objects: int = 3,
        num_pos_x: int = 3,
        num_pos_y: int = 3,
        num_colors: int = 7,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        assert num_samples % max_num_objects == 0, "Samples evenly distriubted among object counts, so num_samples must be divisible by max_num_objects"

        self.num_samples = num_samples
        assert(vsa.num_pos_x == num_pos_x)
        assert(vsa.num_pos_y == num_pos_y)
        assert(vsa.num_colors == num_colors)
        self.vsa = vsa
        self.num_pos_x = num_pos_x
        self.num_pos_y = num_pos_y
        self.num_colors = num_colors
        assert(max_num_objects <= num_pos_x * num_pos_y)
        self.max_num_objects = max_num_objects

        # Generate dataset if not exists
        if force_gen or not self._check_exists(train):
            self.dataset_gen(train)
        else:
            if train:
                for i in range(1, self.max_num_objects+1):
                    self.data += self._load_data(os.path.join(self.root, f"train-images-{i}obj-{num_samples//max_num_objects}samples.pt"))
                    self.labels += self._load_label(os.path.join(self.root, f"train-labels-{i}obj-{num_samples//max_num_objects}samples.json"))
                    self.targets += self._load_data(os.path.join(self.root, f"train-targets-{i}obj-{num_samples//max_num_objects}samples.pt"))
            else:
                self.data = self._load_data(os.path.join(self.root, f"test-images-{num_samples}samples.pt"))
                self.labels = self._load_label(os.path.join(self.root, f"test-labels-{num_samples}samples.json"))
                self.targets = self._load_data(os.path.join(self.root, f"test-targets-{num_samples}samples.pt"))
 
    def _check_exists(self, train: bool) -> bool:
        if (train):
            return all(
                os.path.exists(os.path.join(self.root, file))
                for file in [f"train-images-{n}obj-{self.num_samples//self.max_num_objects}samples.pt" for n in range(1, self.max_num_objects+1)]
            )
        else:
            return all(
                os.path.exists(os.path.join(self.root, file))
                for file in [f"test-images-{self.num_samples}samples.pt"]
            )

    def _load_label(self, path: str):
        with open(path, "r") as path:
            return json.load(path)

    def _load_data(self, path: str):
        return torch.load(path)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)

    def dataset_gen(self, train: bool):
        assert(self.num_colors <= len(self.COLOR_SET))

        num_obj_samples = self.num_samples // self.max_num_objects
        os.makedirs(self.root, exist_ok=True)

        if train:
            raw_train_ds = MNIST(root=os.path.join(self.root, ".."), train=True, download=True)
            for n_obj in range(1, self.max_num_objects+1):
                image_set, label_set = self.data_gen(num_obj_samples, n_obj, raw_train_ds)
                torch.save(image_set, os.path.join(self.root, f"train-images-{n_obj}obj-{num_obj_samples}samples.pt"))
                with open(os.path.join(self.root, f"train-labels-{n_obj}obj-{num_obj_samples}samples.json"), "w") as f:
                    json.dump(label_set, f)
                target_set = self.target_gen(label_set)
                torch.save(target_set, os.path.join(self.root, f"train-targets-{n_obj}obj-{num_obj_samples}samples.pt"))
                self.data += image_set
                self.labels += label_set
                self.targets += target_set
        else:
            # I assume training and testing set are mutually exclusive
            raw_test_ds = MNIST(root=os.path.join(self.root, ".."), train=False, download=True)
            image_set = []
            label_set = []
            for n_obj in range(1, self.max_num_objects+1):
                _image_set, _label_set = self.data_gen(num_obj_samples, n_obj, raw_test_ds)
                image_set += _image_set
                label_set += _label_set
            torch.save(image_set, os.path.join(self.root, f"test-images-{self.num_samples}samples.pt"))
            with open(os.path.join(self.root, f"test-labels-{self.num_samples}samples.json"), "w") as f:
                json.dump(label_set, f)
            target_set = self.target_gen(label_set)
            torch.save(target_set, os.path.join(self.root, f"test-targets-{self.num_samples}samples.pt"))
            self.data = image_set
            self.labels = label_set
            self.targets = target_set


    def data_gen(self, num_samples, num_objs, raw_ds: MNIST) -> [list, list]:
        image_set = []
        label_set_uni = set() # Check the coverage of generation
        label_set = []

        for i in range(num_samples):
            # num_pos_x by num_pos_y grid, each grid 28x28 pixels, color (3-dim uint8 tuple)
            # [H, W, C]
            image_tensor = torch.zeros(28*self.num_pos_y, 28*self.num_pos_x, 3, dtype=torch.uint8)
            label = []
            label_uni = [] # Check the coverage of generation
            # one object max per position
            pos = set()
            for j in range(num_objs):
                while True:
                    pos_x = random.randint(0, self.num_pos_x-1)
                    pos_y = random.randint(0, self.num_pos_y-1)
                    if (pos_x, pos_y) not in pos:
                        pos.add((pos_x, pos_y))
                        break
                color_idx = random.randint(0, self.num_colors-1)
                mnist_idx = random.randint(0, 9999) # 10k images in MNIST test set
                digit_image = raw_ds.data[mnist_idx, :, :]
                for k in self.COLOR_SET[color_idx]:
                    image_tensor[pos_y*28:(pos_y+1)*28, pos_x*28:(pos_x+1)*28, k] = digit_image
                
                object = {
                    "pos_x": pos_x, 
                    "pos_y": pos_y,
                    "color": color_idx,
                    "digit": raw_ds.targets[mnist_idx].item()
                }

                label.append(object)
                # For coverage check. Since pos is checked to be unique, objects are unique
                label_uni.append((pos_x, pos_y, color_idx, raw_ds.targets[mnist_idx].item()))
                
            # For coverage check, convert to tuple to make it hashable and unordered
            label_set_uni.add(tuple(label_uni)) 
            label_set.append(label)
            image_set.append(image_tensor)

        print(f"Generated {num_samples} samples of {num_objs}-object images. Coverage: {len(label_set_uni)} unique labels.")
        return image_set, label_set

    def target_gen(self, label_set: list or str) -> list:
        if type(label_set) == str:
            with open(label_set, "r") as f:
                label_set = json.load(f)
        
        target_set = []
        for label in label_set:
            # Key is a list of tuples of (pos_x, pos_y, color, digit)
            # The order of objects in the list does not matter as they produce the same vector
            key = []
            for i in range(len(label)):
                key.append((label[i]["pos_x"], label[i]["pos_y"], label[i]["color"], label[i]["digit"]))
            
            target_set.append(self.vsa[key])
        return target_set
        
# %%
