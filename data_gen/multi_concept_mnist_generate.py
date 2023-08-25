import torchvision
import random
import matplotlib.pyplot as plt
import torch
import json

dataset = torchvision.datasets.MNIST(root="./mnist_raw", train=False, download=True)

full_label_set = {}

for i in range(100):
    rand_count = random.randint(1, 3)
    image_tensor = torch.zeros(28*3, 28*3, 3, dtype=torch.uint8)
    labels = []
    pos = set()
    for j in range(rand_count):
        loc_x = random.randint(0, 2)
        loc_y = random.randint(0, 2)
        while (loc_x, loc_y) in pos:
            loc_x = random.randint(0, 2)
            loc_y = random.randint(0, 2)
        pos.add((loc_x, loc_y))
        color_idx = random.randint(0, 2)
        mnist_idx = random.randint(0, 9999)
        image = dataset.data[mnist_idx, :, :]
        image_tensor[loc_x*28:(loc_x+1)*28, loc_y*28:(loc_y+1)*28, color_idx] = image
        label = {
            "x": loc_x, 
            "y": loc_y,
            "color": color_idx,
            "digit": dataset.targets[mnist_idx].item()
        }
        labels.append(label)

    torch.save(image_tensor, f"./multi_concept_mnist_dataset/{i}.pt")
    full_label_set[i] = labels

with open("./multi_concept_mnist_dataset/index.json", "w") as f:
    json.dump(full_label_set, f)
    

    