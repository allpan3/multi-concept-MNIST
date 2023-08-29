
import torch
from const import *
from model.vsa import MultiConceptMNISTVSA
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

def collate_fn(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    labels = [x[1] for x in batch]
    targets = torch.stack([x[2] for x in batch], dim=0)
    return imgs, labels, targets


def get_train_test_dls():
    data_dir = f"./data/multi-concept-MNIST/{DIM}dim-{NUM_POS_X}x{NUM_POS_Y}y-{NUM_COLOR}color"
    vsa = MultiConceptMNISTVSA(data_dir, dim=DIM, num_colors=NUM_COLOR, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y)
    train_ds = MultiConceptMNIST(data_dir, vsa, train=True, num_samples=9000, max_num_objects=MAX_NUM_OBJECTS, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)
    test_ds = MultiConceptMNIST(data_dir, vsa, train=False, num_samples=NUM_TEST_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)
    train_ld = DataLoader(train_ds, batch_size=20, shuffle=True, collate_fn=collate_fn)
    test_ld = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return train_ld, test_ld, vsa


def get_model_loss_optimizer():
    model = MultiConceptNonDecomposed(dim=DIM)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if torch.cuda.is_available():
        model = model.cuda()
    return model, loss_fn, optimizer

def train(dataloader, model, loss_fn, optimizer, num_epoch=3):
    writer = SummaryWriter()
    # images in tensor([B, H, W, C])
    # labels in [{'pos_x': tensor, 'pos_y': tensor, 'color': tensor, 'digit': tensor}, ...]
    # targets in VSATensor([B, D])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for epoch in range(num_epoch):
        for idx, (images, _, targets) in enumerate(tqdm(dataloader, desc="train")):
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


def factorization(codebooks, input) -> list:
    max_iters = 50

    n = len(codebooks)
    guesses = [None] * n
    result_set = []
    convergence = []
    # 3 objects max, so we need to run 3 times
    for k in range(MAX_NUM_OBJECTS):
        for i in range(n):
            # guesses[i] = hd.multiset(codebooks[i])
            # guesses[i] = hd.multiset(codebooks[i]).sign()
            guesses[i] = hd.hard_quantize(hd.multiset(codebooks[i]))

        similarities = [None] * n
        old_similarities = None
        for j in range(max_iters):
            estimates = torch.stack(guesses)
            
            inv_estimates = estimates.inverse()

            rolled = []
            for i in range(1, n):
                rolled.append(inv_estimates.roll(i, dims=0))

            inv_estimates = torch.stack(rolled, dim=1)
            others = hd.multibind(inv_estimates)
            # new_estimates = hd.bind(input.sign(), others)
            new_estimates = hd.bind(hd.hard_quantize(input), others)
            # new_estimates = hd.bind(input, others)
            for i in range(0, n):
                similarities[i] = hd.dot_similarity(new_estimates[i], codebooks[i])
                similarities[i] = torch.abs(similarities[i])
                guesses[i] = hd.dot_similarity(similarities[i], codebooks[i].transpose(0,1)).sign()
                # guesses[i] = hd.dot_similarity(similarities[i], codebooks[i].transpose(0,1))
                # guesses[i][guesses[i] > DIM] = DIM
                # guesses[i][guesses[i] < -DIM] = -DIM

            if (old_similarities is not None and all(chain.from_iterable((old_similarities[i] == similarities[i]).tolist() for i in range(n)))):
                break
            
            # TODO sometimes the network reaches a metastable state where the guesses are flipping bits every iteration but not converging
            # should be able to break out of the loop ealier
            old_similarities = similarities.copy()
            # old_estimates = new_estimates
        
        convergence.append(j)

        idx = [None] * n
        output = []
        for i in range(n):
            # idx[i] = torch.argmax(torch.abs(similarities[i])).item()
            idx[i] = torch.argmax(similarities[i]).item()
            output.append(codebooks[i][idx[i]])
        output = torch.stack(output).squeeze()

        # Subtract the object from the inference result
        object = hd.multibind(output)
        input = input - object

        result = {
            'pos_x': idx[0],
            'pos_y': idx[1],
            'color': idx[2],
            'digit': idx[3]
        }

        result_set.append(result)

    return result_set, convergence

if __name__ == "__main__":
    train_dl, test_dl, vsa= get_train_test_dls()
    model, loss_fn, optimizer = get_model_loss_optimizer()
    train(train_dl, model, loss_fn, optimizer)
    exit()

    # Inference
    # images in tensor([B, H, W, C])
    # labels in [{'pos_x': tensor, 'pos_y': tensor, 'color': tensor, 'digit': tensor}, ...]
    # targets in VSATensor([B, D])

    incorrect_count = [0] * MAX_NUM_OBJECTS
    codebooks = [vsa.pos_x, vsa.pos_y, vsa.color, vsa.digit]

    n = 0
    for images, labels, targets in tqdm(test_dl, desc="Test"):
        print(Fore.BLUE + "Test {}".format(n) + Fore.RESET)
        # plt.figure()
        # plt.imshow(images[0])
        # plt.show()
        # print()

        # TODO Add inference step

        infer_result = hd.random(1, dimensions=DIM)  # TODO replace with inference result
        
        print("Similiarty(inference, target) = {}".format(hd.dot_similarity(infer_result, targets[0]).item()))
        
        # Factorization
        # TODO Swap targets[0] with inference result
        result, convergence = factorization(codebooks, targets[0])

        # Compare result
        incorrect = False
        message = ""
        for label in labels[0]:
            # For n objects, only check the first n results
            if (label not in result[0: len(labels[0])+1]):
                message += Fore.RED + "Object {} is not detected.".format(label) + Fore.RESET + "\n"
                incorrect = True
            else:
                message += "Object {} is correctly detected.".format(label) + "\n"

        if incorrect:
            print(f"Test {n} Failed")
            print("Convergence: {}".format(convergence))
            print(message)
            incorrect_count[len(labels[0])-1] += 1 if incorrect else 0

        n += 1
    
    for i in range(MAX_NUM_OBJECTS):
        print("Incorrect count for {} objects: {}".format(i+1, incorrect_count[i]) + "/ {}".format(NUM_TEST_SAMPLES//3))

       
