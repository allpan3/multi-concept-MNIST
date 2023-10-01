""" Multi-Concept MNIST Dataset
"""

from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets.vision import VisionDataset
import torch
import random
import json
import os
from vsa import VSA
from typing import Callable, Optional
import itertools
from tqdm import tqdm

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

    def __init__(
        self,
        root: str,
        vsa: VSA,
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

        if (train):
            # Allocate more samples for larger object counts for training
            total = sum([x+1 for x in range(max_num_objects)])
            self.num_samples = [round((x+1)/total * num_samples) for x in range(max_num_objects)]
        else:
            # Even sample numbers for testing
            self.num_samples = [num_samples // max_num_objects] * max_num_objects

        assert(vsa.num_pos_x == num_pos_x)
        assert(vsa.num_pos_y == num_pos_y)
        assert(vsa.num_colors == num_colors)
        self.vsa = vsa
        self.num_pos_x = num_pos_x
        self.num_pos_y = num_pos_y
        self.num_colors = num_colors
        assert(max_num_objects <= num_pos_x * num_pos_y)
        self.max_num_objects = max_num_objects
        
        self.data = []
        self.labels = []
        self.targets = []

        type = "train" if train else "test"
        # Generate dataset if not exists
        if force_gen or not self._check_exists(type):
            self.dataset_gen(train)
        else:
            for i in range(0, self.max_num_objects):
                self.data += self._load_data(os.path.join(self.root, f"{type}-images-{i+1}obj-{self.num_samples[i]}samples.pt"))
                self.labels += self._load_label(os.path.join(self.root, f"{type}-labels-{i+1}obj-{self.num_samples[i]}samples.json"))
                self.targets += self._load_data(os.path.join(self.root, f"{type}-targets-{i+1}obj-{self.num_samples[i]}samples.pt"))
 
    def _check_exists(self, type: str) -> bool:
        return all(
            os.path.exists(os.path.join(self.root, file))
            for file in [f"{type}-targets-{n+1}obj-{self.num_samples[n]}samples.pt" for n in range(self.max_num_objects)]
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

        os.makedirs(self.root, exist_ok=True)

        if train:
            raw_ds = MNIST(root=os.path.join(self.root, "../.."), train=True, download=True)
            type = "train"
        else:
            # I assume training and testing set are mutually exclusive
            raw_ds = MNIST(root=os.path.join(self.root, "../.."), train=False, download=True)
            type = "test"
        
        for n in range(self.max_num_objects):
            n_obj = n + 1
            image_set, label_set = self.data_gen(self.num_samples[n], n_obj, raw_ds)
            print("Saving images...", end="", flush=True)
            torch.save(image_set, os.path.join(self.root, f"{type}-images-{n_obj}obj-{self.num_samples[n]}samples.pt"))
            print("Done. Saving labels...", end="", flush=True)
            with open(os.path.join(self.root, f"{type}-labels-{n_obj}obj-{self.num_samples[n]}samples.json"), "w") as f:
                json.dump(label_set, f)
            print("Done.")
            target_set = self.target_gen(label_set)
            print("Saving targets...", end="", flush=True)
            torch.save(target_set, os.path.join(self.root, f"{type}-targets-{n_obj}obj-{self.num_samples[n]}samples.pt"))
            print("Done.")
            self.data += image_set
            self.labels += label_set
            self.targets += target_set

    def data_gen(self, num_samples, num_objs, raw_ds: MNIST) -> [list, list]:
        image_set = []
        label_set_uni = set() # Check the coverage of generation
        label_set = []

        for i in tqdm(range(num_samples), desc=f"Generating {num_samples} samples of {num_objs}-object images", leave=False):
            # num_pos_x by num_pos_y grid, each grid 28x28 pixels, color (3-dim uint8 tuple)
            # [H, W, C]
            image_tensor = torch.zeros(28*self.num_pos_y, 28*self.num_pos_x, 3, dtype=torch.uint8)
            label = []
            label_uni = set() # Check the coverage of generation

            # one object max per position
            all_pos = [x for x in itertools.product(range(self.num_pos_x), range(self.num_pos_y))]
            for j in range(num_objs):
                pick = random.randint(0,len(all_pos)-1)
                pos_x, pos_y = all_pos.pop(pick)
                color_idx = random.randint(0, self.num_colors-1)
                mnist_idx = random.randint(0, 59999 if raw_ds.train else 9999)
                digit_image = raw_ds.data[mnist_idx, :, :]
                for k in self.COLOR_SET[color_idx]:
                    image_tensor[pos_y*28:(pos_y+1)*28, pos_x*28:(pos_x+1)*28, k] = digit_image
                digit = raw_ds.targets[mnist_idx].item()
                
                object = {
                    "pos_x": pos_x, 
                    "pos_y": pos_y,
                    "color": color_idx,
                    "digit": digit
                }

                label.append(object)
                # For coverage check. Since pos is checked to be unique, objects are unique
                label_uni.add((pos_x, pos_y, color_idx, digit))

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
        for label in tqdm(label_set, desc=f"Generating targets for {len(label_set[0])}-object images", leave=False):
            target_set.append(self.vsa.lookup(label))
        return target_set
 