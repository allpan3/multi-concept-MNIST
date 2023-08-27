# %%
from torchvision import transforms
from torchvision.datasets import MNIST, utils
from torchvision.datasets.vision import VisionDataset
import torch.utils.data as data
import torch
import random
import json
import os

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
        num_samples: int = 100,
        num_pos_x: int = 3,
        num_pos_y: int = 3,
        num_colors: int = 7
    ) -> None:
        super().__init__(root)

        # Generate VSA

        # Generate dataset if not exists
        if not self._check_exists(num_samples):
            self.data_gen(num_samples, num_pos_x, num_pos_y, num_colors)


    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_exists(self, num_samples) -> bool:
        return all(
            utils.check_integrity(os.path.join(self.raw_folder, file))
            for file in [f"images-{num_samples}.pt", f"labels-{num_samples}.json"]
        )

    def data_gen(self, num_samples, num_pos_x, num_pos_y, num_colors):
        COLOR_SET = [
            range(0,3),   # white
            range(0,1),   # red
            range(1,2),   # green
            range(2,3),   # blue
            range(0,2),   # yellow
            range(0,3,2), # magenta
            range(1,3)    # cyan
        ]

        assert(num_colors <= len(COLOR_SET))
        raw_ds = MNIST(root=os.path.join(self.root, ".."), train=False, download=True)

        label_set = []
        image_set = []
        os.makedirs(self.root, exist_ok=True)

        for i in range(num_samples):
            rand_count = random.randint(1, 3) # number of objects in the image

            # num_pos_x by num_pos_y grid, each grid 28x28 pixels, color (3-dim uint8 tuple)
            image_tensor = torch.zeros(28*num_pos_x, 28*num_pos_y, 3, dtype=torch.uint8)
            label = []
            # one object max per position
            pos = set()
            for j in range(rand_count):
                loc_x = random.randint(0, num_pos_x-1)
                loc_y = random.randint(0, num_pos_y-1)
                while (loc_x, loc_y) in pos:
                    loc_x = random.randint(0, num_pos_x-1)
                    loc_y = random.randint(0, num_pos_y-1)
                pos.add((loc_x, loc_y))
                color_idx = random.randint(0, num_colors-1)
                mnist_idx = random.randint(0, 9999)
                digit_image = raw_ds.data[mnist_idx, :, :]
                for k in COLOR_SET[color_idx]:
                    image_tensor[loc_x*28:(loc_x+1)*28, loc_y*28:(loc_y+1)*28, k] = digit_image
                
                object = {
                    "x": loc_x, 
                    "y": loc_y,
                    "color": color_idx,
                    "digit": raw_ds.targets[mnist_idx].item()
                }
                label.append(object)
                
            label_set.append(label)
            image_set.append(image_tensor)

        torch.save(image_set, os.path.join(self.root, f"images-{num_samples}.pt"))
        with open(os.path.join(self.root, f"labels-{num_samples}.json"), "w") as f:
            json.dump(label_set, f)

        self.target_gen(label_set, num_samples, num_pos_x, num_pos_y, num_colors)


    def target_gen(self, label_set: list or str, num_samples, num_pos_x, num_pos_y, num_colors):
        print("Generating targets...")
        if type(label_set) == str:
            with open(label_set, "r") as f:
                label_set = json.load(f)
        elif type(label_set) == list:
            pass

        for label in label_set:
            count = len(label)
            # for i in range(count):
                # 
# %%
