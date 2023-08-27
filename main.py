
import torch
from const import *
from model.vsa import MultiConceptMNISTVSA
from dataset import MultiConceptMNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from model.nn_non_decomposed import MultiConceptNonDecomposed

DIM = 100

def collate_fn(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    labels = [x[1] for x in batch]
    targets = torch.stack([x[2] for x in batch], dim=0)
    return imgs, labels, targets


def get_train_test_dls():
    vsa = MultiConceptMNISTVSA(dim=DIM, num_colors=NUM_COLOR, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y)
    train_ds = MultiConceptMNIST("./data/multi-concept-MNIST", vsa, train=True, num_samples=9000, max_num_objects=3, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)
    test_ds = MultiConceptMNIST("./data/multi-concept-MNIST", vsa, train=False, num_samples=3, max_num_objects=3, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)
    train_ld = DataLoader(train_ds, batch_size=20, shuffle=True, collate_fn=collate_fn)
    test_ld = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return train_ld, test_ld


def get_model_loss_optimizer():
    model = MultiConceptNonDecomposed(dim=DIM)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, loss_fn, optimizer

def train(dataloader, model, loss_fn, optimizer):
    # images in tensor([B, H, W, C])
    # labels in [{'pos_x': tensor, pos_y: tensor, color: tensor, digit: tensor}, ...]
    # targets in VSATensor([B, D])
    for images, _, targets in tqdm(dataloader, desc="train"):
        images_nchw = (images.type(torch.float32)/255).permute(0,3,1,2)
        targets_float = targets.type(torch.float32)
        output = model(images_nchw)
        loss = loss_fn(output, targets_float)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)


if __name__ == "__main__":
    train_dl, test_dl = get_train_test_dls()
    model, loss_fn, optimizer = get_model_loss_optimizer()
    train(train_dl, model, loss_fn, optimizer)
