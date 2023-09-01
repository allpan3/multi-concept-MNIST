
import torch
from model.vsa import MultiConceptMNISTVSA
from model.resonator import Resonator
from dataset import MultiConceptMNIST
from torch.utils.data import DataLoader
import torchhd as hd
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from model.nn_non_decomposed import MultiConceptNonDecomposed
from itertools import chain
from colorama import Fore
from torch.utils.tensorboard import SummaryWriter
import sys
import os
from datetime import datetime
from pytz import timezone

TEST_BATCH_SIZE = 1
VERBOSE = 1
NUM_ITERATIONS = 100
DIM = 1000
NUM_TEST_SAMPLES = 400
MAX_NUM_OBJECTS = 2
NUM_POS_X = 3
NUM_POS_Y = 3
NUM_COLOR = 3

def collate_fn(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    labels = [x[1] for x in batch]
    targets = torch.stack([x[2] for x in batch], dim=0)
    return imgs, labels, targets


def get_train_test_dls(device = "cpu"):
    data_dir = f"./data/{DIM}dim-{NUM_POS_X}x-{NUM_POS_Y}y-{NUM_COLOR}color"
    vsa = MultiConceptMNISTVSA(data_dir, dim=DIM, num_colors=NUM_COLOR, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, seed=None, device=device)
    train_ds = MultiConceptMNIST(data_dir, vsa, train=True, num_samples=24000, max_num_objects=MAX_NUM_OBJECTS, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)
    test_ds = MultiConceptMNIST(data_dir, vsa, train=False, num_samples=NUM_TEST_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)
    train_ld = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_ld = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return train_ld, test_ld, vsa


def get_model_loss_optimizer():
    model = MultiConceptNonDecomposed(dim=DIM)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if torch.cuda.is_available():
        model.cuda()
    return model, loss_fn, optimizer

def train(dataloader, model, loss_fn, optimizer, num_epoch=1):
    writer = SummaryWriter()
    # images in tensor([B, H, W, C])
    # labels in [{'pos_x': tensor, 'pos_y': tensor, 'color': tensor, 'digit': tensor}, ...]
    # targets in VSATensor([B, D])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for epoch in range(num_epoch):
        for idx, (images, labels, targets) in enumerate(tqdm(dataloader, desc="train")):
            images = images.to(device)
            images_nchw = (images.type(torch.float32)/255).permute(0,3,1,2)
            targets_float = targets.type(torch.float32).to(device)
            output = model(images_nchw)
            loss = loss_fn(output, targets_float)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
            writer.add_scalar('Loss/train', loss, epoch * len(dataloader) + idx)
        # import pdb; pdb.set_trace()


def gen_init_estimates(codebooks: hd.VSATensor or list, batch_size) -> hd.VSATensor:
    if (type(codebooks) == list):
        guesses = [None] * len(codebooks)
        for i in range(len(codebooks)):
            guesses[i] = hd.multiset(codebooks[i])
        init_estimates = torch.stack(guesses)
    else:
        init_estimates = hd.multiset(codebooks)
    
    return init_estimates.unsqueeze(0).repeat(batch_size,1,1)

def factorization(resonator_network, inputs, init_estimates):
    
    inputs = inputs.clone()

    result_set = [[] for _ in range(inputs.size(0))]
    converg_set = [[] for _ in range(inputs.size(0))]
    # Always try to extract MAX_NUM_OBJECTS objects
    for k in range(MAX_NUM_OBJECTS):
        # Run resonator network
        result, convergence = resonator_network(inputs, init_estimates)

        # Split batch results
        for i in range(len(result)):
            result_set[i].append(
                {
                    'pos_x': result[i][0],
                    'pos_y': result[i][1],
                    'color': result[i][2],
                    'digit': result[i][3]
                }
            )
            converg_set[i].append(convergence)

            # Get the object vector and subtract it from the input
            # Key must be a list of tuple
            object = resonator_network.vsa[[result[i]]]
            inputs[i] = inputs[i] - object

    return result_set, converg_set

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, test_dl, vsa = get_train_test_dls(device)
    model, loss_fn, optimizer = get_model_loss_optimizer()
    # assume we provided checkpoint path at the end of the command line
    if sys.argv[-1].endswith(".pt") and os.path.exists(sys.argv[-1]):
        checkpoint = torch.load(sys.argv[-1])
        model.load_state_dict(checkpoint)
    else:
        train(train_dl, model, loss_fn, optimizer, num_epoch=10)
        cur_time_pst = datetime.now().astimezone(timezone('US/Pacific')).strftime("%m-%d-%H-%M")
        model_weight_loc = f"./data/{DIM}dim-{NUM_POS_X}x{NUM_POS_Y}y-{NUM_COLOR}color/model_weight_{cur_time_pst}.pt"
        torch.save(model.state_dict(), model_weight_loc)

    # Inference
    # images in tensor([B, H, W, C])
    # labels in [{'pos_x': tensor, 'pos_y': tensor, 'color': tensor, 'digit': tensor}, ...]
    # targets in VSATensor([B, D])

    incorrect_count = [0] * MAX_NUM_OBJECTS
    unconverged = [[0,0] for _ in range(MAX_NUM_OBJECTS)]    # [correct, incorrect]

    resonator_network = Resonator(vsa, type="CONCURRENT", norm=False, activation="NONE", iterations=NUM_ITERATIONS, device=device)
    init_estimates = gen_init_estimates(vsa.codebooks, TEST_BATCH_SIZE)

    model.eval()
    n = 0
    for images, labels, targets in tqdm(test_dl, desc="Test", leave=True if VERBOSE >= 1 else False):
        # plt.figure()
        # plt.imshow(images[0])
        # plt.show()
        # print()

        # TODO Add inference step

        images = images.to(device)
        images_nchw = (images.type(torch.float32)/255).permute(0,3,1,2)
        infer_result = model(images_nchw)[0].round().type(torch.int8)
        
        # Factorization
        outcomes, convergence = factorization(resonator_network, infer_result, init_estimates)

        # Compare results
        # Batch: multiple samples
        for i in range(len(labels)):

            incorrect = False
            message = ""
            label = labels[i]
            # Sample: multiple objects
            for j in range(len(label)):
                # Incorrect if one object is not detected 
                # For n objects, only check the first n results
                if (label[j] not in outcomes[i][0: len(label)+1]):
                    message += Fore.RED + "Object {} is not detected.".format(label[j]) + Fore.RESET + "\n"
                    incorrect = True
                    unconverged[len(label)-1][1] += 1 if convergence[i][j] == NUM_ITERATIONS-1 else 0
                else:
                    message += "Object {} is correctly detected.".format(label[j]) + "\n"
                    unconverged[len(label)-1][0] += 1 if convergence[i][j] == NUM_ITERATIONS-1 else 0

            if incorrect:
                incorrect_count[len(label)-1] += 1 if incorrect else 0
                if (VERBOSE >= 1):
                    print(Fore.BLUE + f"Test {n} Failed:      Convergence = {convergence[i]}" + Fore.RESET)
                    print("Inference result similarity = {:.4f}".format(hd.cosine_similarity(infer_result[i], targets[i]).item()))
                    print(message[:-1])
            else:
                if (VERBOSE >= 2):
                    print(Fore.BLUE + f"Test {n} Passed:      Convergence = {convergence[i]}" + Fore.RESET)
                    print("Inference result similarity = {:.4f}".format(hd.cosine_similarity(infer_result[i], targets[i]).item()))
                    print(message[:-1])
            n += 1

    for i in range(MAX_NUM_OBJECTS):
        print(f"{i+1} objects: Accuracy = {NUM_TEST_SAMPLES//MAX_NUM_OBJECTS - incorrect_count[i]}/{NUM_TEST_SAMPLES//MAX_NUM_OBJECTS}     Unconverged = {unconverged[i]}/{NUM_TEST_SAMPLES//MAX_NUM_OBJECTS}")


       
