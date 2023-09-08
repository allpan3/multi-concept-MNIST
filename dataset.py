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
from model.vsa import MultiConceptMNISTVSA1, MultiConceptMNISTVSA2
from typing import Callable, Optional

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
            self.num_samples = [0] * max_num_objects
            if max_num_objects == 1:
                self.num_samples[0] = num_samples
            elif max_num_objects == 2:
                self.num_samples[0] = int(num_samples * 0.3)
                self.num_samples[1] = int(num_samples * 0.7)
            elif max_num_objects == 3:
                self.num_samples[0] = int(num_samples * 0.15)
                self.num_samples[1] = int(num_samples * 0.35)
                self.num_samples[2] = int(num_samples * 0.5)
            else:
                raise NotImplementedError
        else:
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
            for file in [f"{type}-images-{n+1}obj-{self.num_samples[n]}samples.pt" for n in range(self.max_num_objects)]
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
            torch.save(image_set, os.path.join(self.root, f"{type}-images-{n_obj}obj-{self.num_samples[n]}samples.pt"))
            with open(os.path.join(self.root, f"{type}-labels-{n_obj}obj-{self.num_samples[n]}samples.json"), "w") as f:
                json.dump(label_set, f)
            target_set = self.target_gen(label_set)
            torch.save(target_set, os.path.join(self.root, f"{type}-targets-{n_obj}obj-{self.num_samples[n]}samples.pt"))
            self.data += image_set
            self.labels += label_set
            self.targets += target_set

    def data_gen(self, num_samples, num_objs, raw_ds: MNIST) -> [list, list]:
        image_set = []
        label_set_uni = set() # Check the coverage of generation
        label_set = []

        for i in range(num_samples):
            # num_pos_x by num_pos_y grid, each grid 28x28 pixels, color (3-dim uint8 tuple)
            # [H, W, C]
            image_tensor = torch.zeros(28*self.num_pos_y, 28*self.num_pos_x, 3, dtype=torch.uint8)
            label = []
            label_uni = set() # Check the coverage of generation

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
                mnist_idx = random.randint(0, 59999 if raw_ds.train else 9999)
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
                label_uni.add((pos_x, pos_y, color_idx, raw_ds.targets[mnist_idx].item()))

            # For coverage check, convert to tuple to make it hashable and unordered
            label_set_uni.add(tuple(label_uni)) 
            label_set.append(label)
            image_set.append(image_tensor)

        print(f"Generated {num_samples} samples of {num_objs}-object images. Coverage: {len(label_set_uni)} unique labels.")
        return image_set, label_set  

    def target_gen(self, label_set: list) -> list:
        raise NotImplementedError
 
        
class MultiConceptMNIST1(MultiConceptMNIST):
    def __init__(
        self,
        root: str,
        vsa: MultiConceptMNISTVSA1,
        train: bool,      # training set or test set
        num_samples: int,
        force_gen: bool = False,  # generate dataset even if it exists
        max_num_objects: int = 3,
        num_pos_x: int = 3,
        num_pos_y: int = 3,
        num_colors: int = 7
    ) -> None:
        super().__init__(root, vsa, train, num_samples, force_gen, max_num_objects, num_pos_x, num_pos_y, num_colors)   

    def target_gen(self, label_set: list) -> list:
        if type(label_set) == str:
            with open(label_set, "r") as f:
                label_set = json.load(f)
        
        target_set = []
        for label in label_set:
            target_set.append(self.vsa.lookup(label))
        return target_set


class MultiConceptMNIST2(MultiConceptMNIST):
    def __init__(
        self,
        root: str,
        vsa: MultiConceptMNISTVSA2,
        train: bool,      # training set or test set
        num_samples: int,
        force_gen: bool = False,  # generate dataset even if it exists
        max_num_objects: int = 3,
        num_pos_x: int = 3,
        num_pos_y: int = 3,
        num_colors: int = 7
    ) -> None:
        super().__init__(root, vsa, train, num_samples, force_gen, max_num_objects, num_pos_x, num_pos_y, num_colors)

    def target_gen(self, label_set: list) -> list:
        if type(label_set) == str:
            with open(label_set, "r") as f:
                label_set = json.load(f)
        
        target_set = []
        for label in label_set:
            objects = self.vsa.lookup(label, with_id=False)
            # For n objects in the label, only combine the first n IDs
            ids = self.vsa.multiset(self.vsa.id_codebook[0:len(label)])
            target_set.append(self.vsa.bind(ids, objects))
        return target_set
