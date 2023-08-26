# %%
import torchvision
import random
import torch
import json
import os
import matplotlib.pyplot as plt

dataset = torchvision.datasets.MNIST(root="./mnist_raw", train=False, download=True)

NUM_POS_X = 3
NUM_POS_Y = 3
NUM_COLOR = 7

COLOR_SET = [
    range(0,3),   # white
    range(0,1),   # red
    range(1,2),   # green
    range(2,3),   # blue
    range(0,2),   # yellow
    range(0,3,2), # magenta
    range(1,3)    # cyan
]

full_label_set = {}
os.makedirs("./multi_concept_mnist_dataset", exist_ok=True)

for i in range(1):
    rand_count = random.randint(1, 3)
    # 28x28 images, 3x3 grid, color (3-dim tuple)
    image_tensor = torch.zeros(28*NUM_POS_X, 28*NUM_POS_Y, 3, dtype=torch.uint8)
    labels = []
    pos = set()
    for j in range(rand_count):
        loc_x = random.randint(0, NUM_POS_X-1)
        loc_y = random.randint(0, NUM_POS_Y-1)
        while (loc_x, loc_y) in pos:
            loc_x = random.randint(0, NUM_POS_X-1)
            loc_y = random.randint(0, NUM_POS_Y-1)
        pos.add((loc_x, loc_y))
        color_idx = random.randint(0, NUM_COLOR-1)
        mnist_idx = random.randint(0, 9999)
        digit_image = dataset.data[mnist_idx, :, :]
        for j in COLOR_SET[color_idx]:
            image_tensor[loc_x*28:(loc_x+1)*28, loc_y*28:(loc_y+1)*28, j] = digit_image
        
        label = {
            "x": loc_x, 
            "y": loc_y,
            "color": color_idx,
            "digit": dataset.targets[mnist_idx].item()
        }
        labels.append(label)

    torch.save(image_tensor, f"./multi_concept_mnist_dataset/{i}.pt")
    full_label_set[i] = labels

# %%
with open("./multi_concept_mnist_dataset/index.json", "w") as f:
    json.dump(full_label_set, f)
    

    