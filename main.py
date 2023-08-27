
import os
from const import *
from model.vsa import MultiConceptMNISTVSA
from dataset import MultiConceptMNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
vsa = MultiConceptMNISTVSA(dim=100, num_colors=NUM_COLOR, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y)
train_ds = MultiConceptMNIST("./data/multi-concept-MNIST", vsa, train=True, num_samples=9000, max_num_objects=3, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)
test_ds = MultiConceptMNIST("./data/multi-concept-MNIST", vsa, train=False, num_samples=900, max_num_objects=3, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)

train_ld = DataLoader(train_ds, batch_size=1, shuffle=True)
test_ld = DataLoader(test_ds, batch_size=1, shuffle=False)

# for samples, labels in tqdm(train_ld, desc="Training"):

