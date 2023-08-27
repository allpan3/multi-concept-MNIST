# %%
import os
from const import *
from model.vsa import MultiConceptMNISTVSA
from dataset.dataset import MultiConceptMNIST

vsa = MultiConceptMNISTVSA(dim=100, num_colors=NUM_COLOR, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y)
ds = MultiConceptMNIST("./data/multi-concept-MNIST", vsa, num_train_samples=9000, num_test_samples=900, max_num_objects=3, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)



