
import os
from const import *
from model.vsa import MultiConceptMNISTVSA
from dataset import MultiConceptMNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

vsa = MultiConceptMNISTVSA(dim=100, num_colors=NUM_COLOR, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y)
train_ds = MultiConceptMNIST("./data/multi-concept-MNIST", vsa, train=True, num_samples=9000, max_num_objects=3, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)
test_ds = MultiConceptMNIST("./data/multi-concept-MNIST", vsa, train=False, num_samples=3, max_num_objects=3, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)

train_ld = DataLoader(train_ds, batch_size=1, shuffle=True)
test_ld = DataLoader(test_ds, batch_size=1, shuffle=False)


# images in tensor([B, H, W, C])
# labels in [{'pos_x': tensor, pos_y: tensor, color: tensor, digit: tensor}, ...]
# targets in VSATensor([B, D])
for images, labels, targets in tqdm(test_ld, desc="Test"):
    plt.figure()
    plt.imshow(images[0])
    print(labels)
    print(targets)
    print()

plt.show()
