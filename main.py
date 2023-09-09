
import torch
from model.vsa import MultiConceptMNISTVSA1, MultiConceptMNISTVSA2
from vsa import Resonator
from dataset import MultiConceptMNIST
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from model.nn_non_decomposed import MultiConceptNonDecomposed
from itertools import chain
from colorama import Fore
from torch.utils.tensorboard import SummaryWriter
import sys
import os
from datetime import datetime
from pytz import timezone

VERBOSE = 1
RUN_MODE = "TEST" # "TRAIN", "TEST", "DATAGEN"
ALGO = "algo2" # "algo1", "algo2"
VSA_MODE = "HARDWARE" # "SOFTWARE", "HARDWARE"
DIM = 2000
MAX_NUM_OBJECTS = 2
NUM_POS_X = 3
NUM_POS_Y = 3
NUM_COLOR = 3
# Train
TRAIN_EPOCH = 50
TRAIN_BATCH_SIZE = 128
NUM_TRAIN_SAMPLES = 48000
# Test
TEST_BATCH_SIZE = 1
NUM_TEST_SAMPLES = 300
# Resonator
NORMALIZE = True    # Only applies to SOFTWARE mode. This controls the normalization of the input and estimate vectors to the resonator network
ACTIVATION = "NONE"  # "NONE", "ABS", "NONNEG"
RESONATOR_TYPE = "SEQUENTIAL" # "SEQUENTIAL", "CONCURRENT"
NUM_ITERATIONS = 1000
if VSA_MODE == "HARDWARE":
    NORMALIZE = True # Normalization is forced to be applied in hardware mode

test_dir = f"./tests/{VSA_MODE}-{DIM}dim-{MAX_NUM_OBJECTS}obj-{NUM_POS_X}x-{NUM_POS_Y}y-{NUM_COLOR}color/{ALGO}"

def collate_fn(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    labels = [x[1] for x in batch]
    targets = torch.stack([x[2] for x in batch], dim=0)
    return imgs, labels, targets

def train(dataloader, model, loss_fn, optimizer, num_epoch, cur_time, device = "cpu"):
    writer = SummaryWriter(log_dir=f"./runs/{cur_time}-{ALGO}-{VSA_MODE}-{DIM}dim-{MAX_NUM_OBJECTS}objs-{NUM_POS_X}x-{NUM_POS_Y}y-{NUM_COLOR}color", filename_suffix=f".{TRAIN_BATCH_SIZE}batch-{TRAIN_EPOCH}epoch-{NUM_TRAIN_SAMPLES}samples")
    # images in tensor([B, H, W, C])
    # labels in [{'pos_x': tensor, 'pos_y': tensor, 'color': tensor, 'digit': tensor}, ...]
    # targets in VSATensor([B, D])
    for epoch in range(num_epoch):
        for idx, (images, labels, targets) in enumerate(tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)):
            images = images.to(device)
            images_nchw = (images.type(torch.float32)/255).permute(0,3,1,2)
            targets_float = targets.to(device).type(torch.float32)
            output = model(images_nchw)
            loss = loss_fn(output, targets_float)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
            writer.add_scalar('Loss/train', loss, epoch * len(dataloader) + idx)
        # import pdb; pdb.set_trace()

    return round(loss.item(), 4)


def get_similarity(v1, v2):
    """
    Return the hamming similarity for normalized vectors, and cosine similarity for unnormalized vectors
    Hamming similarity is linear and should reflect the noise level
    Cosine similarity is non-linear and may not reflect the noise level
    """
    if VSA_MODE == "SOFTWARE":
        if NORMALIZE:
            # Compare the normalized vectors
            positive = torch.tensor(1, device=v1.device)
            negative = torch.tensor(-1, device=v1.device)
            v1_ = torch.where(v1 >= 0, positive, negative)
            v2_ = torch.where(v2 >= 0, positive, negative)
            return torch.sum(torch.where(v1_ == v2_, 1, 0), dim=-1) / DIM
        else:
            v1_dot = torch.sum(v1 * v1, dim=-1)
            v1_mag = torch.sqrt(v1_dot)
            v2_dot = torch.sum(v2 * v2, dim=-1)
            v2_mag = torch.sqrt(v2_dot)
            magnitude = v1_mag * v2_mag
            magnitude = torch.clamp(magnitude, min=1e-08)
            return torch.matmul(v1.type(torch.float32), v2.type(torch.float32)) / magnitude
    else:
        return torch.sum(torch.where(v1 == v2, 1, 0), dim=-1) / DIM

def get_vsa():
    if ALGO == "algo1":
        vsa = MultiConceptMNISTVSA1(test_dir, model=VSA_MODE, dim=DIM, max_num_objects=MAX_NUM_OBJECTS, num_colors=NUM_COLOR, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, seed=0, device=device)
    elif ALGO == "algo2":
        vsa = MultiConceptMNISTVSA2(test_dir, model=VSA_MODE, dim=DIM, max_num_objects=MAX_NUM_OBJECTS, num_colors=NUM_COLOR, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, seed=0, device=device)
    return vsa


def get_train_data(vsa):
    train_ds = MultiConceptMNIST(test_dir, vsa, train=True, num_samples=NUM_TRAIN_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)
    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return train_dl

def get_test_data(vsa):
    test_ds = MultiConceptMNIST(test_dir, vsa, train=False, num_samples=NUM_TEST_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)
    test_dl = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return test_dl


def test_algo1(vsa, model, test_dl, device):
    """
    Algorithm 1
    The input vector is an un-normalized integer vector. After an object is extracted, it is
    subtracted from the input integer vector. The updated vector goes through the resonator network
    again to extract the next object.
    Since the input vector is expected to be un-normalized (except when there's only one object),
    this algorithm cannot extract more than 1 object when running in hardware mode, as the vectors
    are automatically normalized after bundling. The NN model would be trained with normalized vectors.
        This should be fixable, but may not be useful as it will require the NN model to be trained
        with bipolarized vectors, which is exactly the same as in the software mode. So we can just 
        use the NN model trained in software mode and binarize to 1/0 instead of 1/-1 before factorization
        in real hardware.

    Note we can still normalize the input and estimate vectors (controlled by NORMALIZE flag), but
    the NN model should be trained with un-normalized vectors in this algorithm.
    """

    def factorization(vsa, resonator_network, inputs, init_estimates, codebooks = None, orig_indices = None):
        # The input vector is an integer vector. Subtract every object vector from the input vector
        # and feed to the resonator netowrk again to get the next vector
        result_set = [[] for _ in range(inputs.size(0))]
        converg_set = [[] for _ in range(inputs.size(0))]

        inputs = inputs.clone()

        # Always try to extract MAX_NUM_OBJECTS objects
        for k in range(MAX_NUM_OBJECTS):

            if NORMALIZE and VSA_MODE == "SOFTWARE":
                inputs_ = vsa.normalize(inputs)
            else:
                inputs_ = inputs

            # Run resonator network
            result, convergence = resonator_network(inputs_, init_estimates, codebooks, orig_indices)

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

                if VSA_MODE == "SOFTWARE":
                    # Get the object vector and subtract it from the input
                    # Key must be a list of tuple
                    object = vsa[[result[i]]]
                    inputs[i] = inputs[i] - object

        return result_set, converg_set


    rn = Resonator(vsa, type=RESONATOR_TYPE, activation=ACTIVATION, iterations=NUM_ITERATIONS, device=device)

    codebooks, orig_indices = rn.reorder_codebooks()
    init_estimates = rn.get_init_estimates(codebooks, NORMALIZE, TEST_BATCH_SIZE)

    incorrect_count = [0] * MAX_NUM_OBJECTS
    unconverged = [[0,0] for _ in range(MAX_NUM_OBJECTS)]    # [correct, incorrect]
    n = 0

    # images in tensor([B, H, W, C])
    # labels in [{'pos_x': tensor, 'pos_y': tensor, 'color': tensor, 'digit': tensor}, ...]
    # targets in VSATensor([B, D])
    for images, labels, targets in tqdm(test_dl, desc="Test", leave=True if VERBOSE >= 1 else False):
        images = images.to(device)
        images_nchw = (images.type(torch.float32)/255).permute(0,3,1,2)
        infer_result = model(images_nchw)
        if VSA_MODE == "SOFTWARE":
            # round() will round numbers near 0 to 0, which is not ideal when there's one object, since the vector should be bipolar
            # But 0 is legitimate when there are multiple objects.
            infer_result = infer_result.round().type(torch.int8)
        else:
            infer_result = torch.sigmoid(infer_result).round().type(torch.int8)

        # Factorization
        outcomes, convergence = factorization(vsa, rn, infer_result, init_estimates, codebooks, orig_indices)

        # Compare results
        # Batch: multiple samples
        for i in range(len(labels)):
            incorrect = False
            message = ""
            label = labels[i]
            sim_per_obj = []

            # Sample: multiple objects
            for j in range(len(label)):
                # Incorrect if one object is not detected 
                # For n objects, only check the first n results
                if (label[j] not in outcomes[i][0: len(label)]):
                    message += Fore.RED + "Object {} is not detected.".format(label[j]) + Fore.RESET + "\n"
                    incorrect = True
                    unconverged[len(label)-1][1] += 1 if convergence[i][j] == NUM_ITERATIONS-1 else 0
                else:
                    message += "Object {} is correctly detected.".format(label[j]) + "\n"
                    unconverged[len(label)-1][0] += 1 if convergence[i][j] == NUM_ITERATIONS-1 else 0

                # Collect per-object similarity
                sim_per_obj.append(round(get_similarity(infer_result[i], vsa.lookup([label[j]])).item(), 3))

            if incorrect:
                incorrect_count[len(label)-1] += 1 if incorrect else 0
                if (VERBOSE >= 1):
                    print(Fore.BLUE + f"Test {n} Failed:      Convergence = {convergence[i]}" + Fore.RESET)
                    print("Inference result similarity = {:.3f}".format(get_similarity(infer_result[i], targets[i]).item()))
                    print("Per-object similarity = {}".format(sim_per_obj))
                    print(message[:-1])
                    print("Outcome = {}".format(outcomes[i][0: len(label)]))
            else:
                if (VERBOSE >= 2):
                    print(Fore.BLUE + f"Test {n} Passed:      Convergence = {convergence[i]}" + Fore.RESET)
                    print("Inference result similarity = {:.3f}".format(get_similarity(infer_result[i], targets[i]).item()))
                    print("Per-object similarity = {}".format(sim_per_obj))
                    print(message[:-1])
            n += 1

    return incorrect_count, unconverged


def test_algo2(vsa, model, test_dl, device):

    def factorization(vsa, resonator_network, inputs, init_estimates, codebooks, orig_indices = None):
        result_set = [[] for _ in range(inputs.size(0))]
        converg_set = [[] for _ in range(inputs.size(0))]

        inputs = inputs.clone()
        # In hardware mode the input is expected to be normalized already, so shouldn't do it here
        if NORMALIZE and VSA_MODE == "SOFTWARE":
            inputs = vsa.normalize(inputs)

        for k in range(MAX_NUM_OBJECTS):
            # Manually unbind ID 
            inputs_ = vsa.bind(inputs, vsa.id_codebook[k])

            # Run resonator network
            result, convergence = resonator_network(inputs_, init_estimates, codebooks, orig_indices)

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
        return result_set, converg_set

    rn = Resonator(vsa, type=RESONATOR_TYPE, activation=ACTIVATION, iterations=NUM_ITERATIONS, device=device)

    # Remove the ID codebook since it is manually unbound
    codebooks = vsa.codebooks[:-1]

    codebooks, orig_indices = rn.reorder_codebooks(codebooks)
    init_estimates = rn.get_init_estimates(codebooks, NORMALIZE, TEST_BATCH_SIZE)

    incorrect_count = [0] * MAX_NUM_OBJECTS
    unconverged = [[0,0] for _ in range(MAX_NUM_OBJECTS)]    # [correct, incorrect]
    n = 0

    # images in tensor([B, H, W, C])
    # labels in [{'pos_x': tensor, 'pos_y': tensor, 'color': tensor, 'digit': tensor}, ...]
    # targets in VSATensor([B, D])
    for images, labels, targets in tqdm(test_dl, desc="Test", leave=True if VERBOSE >= 1 else False):

        # Inference
        images = images.to(device)
        images_nchw = (images.type(torch.float32)/255).permute(0,3,1,2)
        infer_result = model(images_nchw)
        if VSA_MODE == "SOFTWARE":
            infer_result = infer_result.round().type(torch.int8)
        else:
            infer_result = torch.sigmoid(infer_result).round().type(torch.int8)

        # Factorization
        outcomes, convergence = factorization(vsa, rn, infer_result, init_estimates, codebooks, orig_indices)

        # Compare results
        # Batch: multiple samples
        for i in range(len(labels)):
            incorrect = False
            message = ""
            label = labels[i]
            sim_per_obj = []

            # Must look up the entire label (instead of one object at a time) together to get the correct reordering done
            gt_objs = vsa.lookup(label, bundled=False)

            # Sample: multiple objects
            for j in range(len(label)):
                # Incorrect if one object is not detected 
                # For n objects, only check the first n results
                if (label[j] not in outcomes[i][0: len(label)]):
                    message += Fore.RED + "Object {} is not detected.".format(label[j]) + Fore.RESET + "\n"
                    incorrect = True
                    unconverged[len(label)-1][1] += 1 if convergence[i][j] == NUM_ITERATIONS-1 else 0
                else:
                    message += "Object {} is correctly detected.".format(label[j]) + "\n"
                    unconverged[len(label)-1][0] += 1 if convergence[i][j] == NUM_ITERATIONS-1 else 0
                
                sim_per_obj.append(round(get_similarity(infer_result[i], gt_objs[j]).item(), 3))

            if incorrect:
                incorrect_count[len(label)-1] += 1 if incorrect else 0
                if (VERBOSE >= 1):
                    print(Fore.BLUE + f"Test {n} Failed:      Convergence = {convergence[i]}" + Fore.RESET)
                    print("Inference result similarity = {:.3f}".format(get_similarity(infer_result[i], targets[i]).item()))
                    print("Per-object similarity = {}".format(sim_per_obj))
                    print(message[:-1])
                    print("Outcome = {}".format(outcomes[i][0: len(label)]))
            else:
                if (VERBOSE >= 2):
                    print(Fore.BLUE + f"Test {n} Passed:      Convergence = {convergence[i]}" + Fore.RESET)
                    print("Inference result similarity = {:.3f}".format(get_similarity(infer_result[i], targets[i]).item()))
                    print("Per-object similarity = {}".format(sim_per_obj))
                    print(message[:-1])
            n += 1

    return incorrect_count, unconverged
        

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Workload Config: algorithm = {ALGO}, vsa mode = {VSA_MODE}, dim = {DIM}, num pos x = {NUM_POS_X}, num pos y = {NUM_POS_Y}, num color = {NUM_COLOR}, num digits = 10, max num objects = {MAX_NUM_OBJECTS}")

    vsa = get_vsa()
    model = MultiConceptNonDecomposed(dim=DIM, device=device)

    if RUN_MODE == "TRAIN":
        print(f"Training on {device}: samples = {NUM_TRAIN_SAMPLES}, epochs = {TRAIN_EPOCH}, batch size = 128")
        loss_fn = torch.nn.MSELoss() if VSA_MODE == "SOFTWARE" else torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        cur_time_pst = datetime.now().astimezone(timezone('US/Pacific')).strftime("%m-%d-%H-%M")
        train_dl = get_train_data(vsa)
        final_loss = train(train_dl, model, loss_fn, optimizer, num_epoch=TRAIN_EPOCH, cur_time=cur_time_pst, device=device)
        model_weight_loc = os.path.join(test_dir, f"model_weights_{MAX_NUM_OBJECTS}objs_{TRAIN_BATCH_SIZE}batch_{TRAIN_EPOCH}epoch_{NUM_TRAIN_SAMPLES}samples_{final_loss}loss_{cur_time_pst}.pt")
        torch.save(model.state_dict(), model_weight_loc)
        print(f"Model weights saved to {model_weight_loc}")

    # Test mode
    elif RUN_MODE == "TEST":
        print(f"Running test on {device}, batch size = {TEST_BATCH_SIZE}")

        # assume we provided checkpoint path at the end of the command line
        if sys.argv[-1].endswith(".pt") and os.path.exists(sys.argv[-1]):
            checkpoint = torch.load(sys.argv[-1])
            model.load_state_dict(checkpoint)
        else:
            print("Please provide a valid model checkpoint path.")
            exit(1)

        model.eval()

        print(f"Resonator setup: type = {RESONATOR_TYPE}, normalize = {NORMALIZE}, activation = {ACTIVATION}, iterations = {NUM_ITERATIONS}")

        test_dl = get_test_data(vsa)
 
        if ALGO == "algo1":
            incorrect_count, unconverged = test_algo1(vsa, model, test_dl, device)
        elif ALGO == "algo2":
            incorrect_count, unconverged = test_algo2(vsa, model, test_dl, device)

        for i in range(MAX_NUM_OBJECTS):
            print(f"{i+1} objects: Accuracy = {NUM_TEST_SAMPLES//MAX_NUM_OBJECTS - incorrect_count[i]}/{NUM_TEST_SAMPLES//MAX_NUM_OBJECTS}     Unconverged = {unconverged[i]}/{NUM_TEST_SAMPLES//MAX_NUM_OBJECTS * (i+1)}")

    # Data Gen mode      
    else:
        MultiConceptMNIST(test_dir, vsa, train=True, num_samples=NUM_TRAIN_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR, force_gen=True)
        MultiConceptMNIST(test_dir, vsa, train=False, num_samples=NUM_TEST_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR, force_gen=True)

    

