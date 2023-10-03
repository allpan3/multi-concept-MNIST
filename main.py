
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
###########
# Configs #
###########
VERBOSE = 1
SEED = 0
ALGO = "algo1" # "algo1", "algo2"
VSA_MODE = "HARDWARE" # "SOFTWARE", "HARDWARE"
DIM = 1024
MAX_NUM_OBJECTS = 6
SINGLE_COUNT = False # True, False
NUM_POS_X = 3
NUM_POS_Y = 3
NUM_COLOR = 7
# Hardware config
EHD_BITS = 9
SIM_BITS = 13
# Train
TRAIN_EPOCH = 30
TRAIN_BATCH_SIZE = 256
NUM_TRAIN_SAMPLES = 300000
# Test
TEST_BATCH_SIZE = 1
NUM_TEST_SAMPLES = 600
# Resonator
RESONATOR_TYPE = "SEQUENTIAL" # "SEQUENTIAL", "CONCURRENT"
MAX_TRIALS = MAX_NUM_OBJECTS + 5
NUM_ITERATIONS = 200
ACTIVATION = 'THRESH_AND_SCALE'      # 'IDENTITY', 'THRESHOLD', 'SCALEDOWN', "THRESH_AND_SCALE"
ACT_VALUE = 16
STOCHASTICITY = "SIMILARITY"  # apply stochasticity: "NONE", "SIMILARITY", "VECTOR"
RANDOMNESS = 0.04
# Similarity thresholds are affected by the maximum number of vectors superposed. These values need to be lowered when more vectors are superposed
SIM_EXPLAIN_THRESHOLD = 0.25
SIM_DETECT_THRESHOLD = 0.15
ENERGY_THRESHOLD = 0.25
EARLY_CONVERGE = 0.6
EARLY_TERM_THRESHOLD = 0.15

# In hardware mode, the activation value needs to be a power of two
if VSA_MODE == "HARDWARE" and (ACTIVATION == "SCALEDOWN" or ACTIVATION == "THRESH_AND_SCALE"):
    def biggest_power_two(n):
        """Returns the biggest power of two <= n"""
        # if n is a power of two simply return it
        if not (n & (n - 1)):
            return n
        # else set only the most significant bit
        return int("1" + (len(bin(n)) - 3) * "0", 2)
    ACT_VALUE = biggest_power_two(ACT_VALUE)

# If activation is scaledown, then the early convergence threshold needs to scale down accordingly
if EARLY_CONVERGE is not None and (ACTIVATION == "SCALEDOWN" or ACTIVATION == "THRESHOLDED_SCALEDOWN"):
    EARLY_CONVERGE = EARLY_CONVERGE / ACT_VALUE


test_dir = f"./tests/{VSA_MODE}-{DIM}dim-{NUM_POS_X}x-{NUM_POS_Y}y-{NUM_COLOR}color/{ALGO}"

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
        if epoch != 0 and epoch % 5 == 0:
            model_weight_loc = os.path.join(test_dir, f"model_weights_{MAX_NUM_OBJECTS}objs{'_single_count' if SINGLE_COUNT else ''}_{TRAIN_BATCH_SIZE}batch_{epoch}epoch_{NUM_TRAIN_SAMPLES}samples_{round(loss.item(),4)}loss_{cur_time}.pt")
            torch.save(model.state_dict(), model_weight_loc)
            print(f"Model checkpoint saved to {model_weight_loc}")

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

def get_similarity(v1, v2, quantized):
    """
    Return the hamming similarity.
    Always compare the similarity between quantized vectors. If inputs are not quantized, quantize them first.
    Hamming similarity is linear and should reflect the noise level
    """
    if not quantized:
        if VSA_MODE == "SOFTWARE":
            # Compare the quantized vectors
            v1 = torch.where(v1 >= 0, 1, -1).to(v1.device)
            v2 = torch.where(v2 >= 0, 1, -1).to(v1.device)
        else:
            v1 = torch.where(v1 >= 0, 1, 0).to(v1.device)
            v2 = torch.where(v2 >= 0, 1, 0).to(v1.device)

    return torch.sum(torch.where(v1 == v2, 1, 0), dim=-1) / DIM
            
def get_vsa(device):
    if ALGO == "algo1":
        vsa = MultiConceptMNISTVSA1(test_dir, mode=VSA_MODE, dim=DIM, max_num_objects=MAX_NUM_OBJECTS, num_colors=NUM_COLOR, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, ehd_bits=EHD_BITS, sim_bits=SIM_BITS, seed=SEED, device=device)
    elif ALGO == "algo2":
        vsa = MultiConceptMNISTVSA2(test_dir, mode=VSA_MODE, dim=DIM, max_num_objects=MAX_NUM_OBJECTS, num_colors=NUM_COLOR, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, ehd_bits=EHD_BITS, sim_bits=SIM_BITS, seed=SEED, device=device)
    return vsa

def get_train_data(vsa):
    train_ds = MultiConceptMNIST(test_dir, vsa, train=True, num_samples=NUM_TRAIN_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, single_count=SINGLE_COUNT, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)
    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return train_dl

def get_test_data(vsa):
    test_ds = MultiConceptMNIST(test_dir, vsa, train=False, num_samples=NUM_TEST_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, single_count=SINGLE_COUNT, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR)
    test_dl = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return test_dl


def test_algo1(vsa, model, test_dl, device):
    """
    Algorithm 1: explain away
    See VSA-factorization repo for details
    NN needs to be trained with unquantized vectors
    """

    assert TEST_BATCH_SIZE == 1, "Batch size != 1 is not yet supported"

    def factorization(vsa, rn, inputs, init_estimates):

        outcomes = [[] for _ in range(inputs.size(0))]  # num of batches
        unconverged = [0] * inputs.size(0)
        iters = [[] for _ in range(inputs.size(0))]
        sim_to_remain = [[] for _ in range(inputs.size(0))]
        sim_to_remain_all = [[] for _ in range(inputs.size(0))]
        sim_to_orig = [[] for _ in range(inputs.size(0))]
        sim_to_orig_all = [[] for _ in range(inputs.size(0))]
        debug_message = ""

        inputs = inputs.clone()
        inputs_q = vsa.quantize(inputs)
        init_estimates = init_estimates.clone()

        for k in range(MAX_TRIALS):
            inputs_ = vsa.quantize(inputs)

            # # Apply stochasticity to initial estimates
            # if vsa.mode == "HARDWARE":
            #     # This is one way to randomize the initial estimates
            #     init_estimates = vsa.ca90(init_estimates)
            # elif vsa.mode == "SOFTWARE":
            #     # Replace this with true random vector
            #     init_estimates = vsa.apply_noise(init_estimates, 0.5)

            # Run resonator network
            outcome, iter, converge = rn(inputs_, init_estimates)

            # Split batch results
            for i in range(len(outcome)):
                unconverged[i] += 1 if converge == False else 0
                iters[i].append(iter)
                # Get the compositional vector and subtract it from the input
                vector = vsa.get_vector(outcome[i])
                sim_orig = vsa.dot_similarity(inputs_q[i], vector)
                sim_remain = vsa.dot_similarity(inputs_[i], vector)
                explained = "NOT EXPLAINED"
                sim_to_orig_all[i].append(sim_orig)
                sim_to_remain_all[i].append(sim_remain)
                # Only explain away the vector if it's similar enough to the input
                # Also only consider it as the final candidate if so
                if sim_remain >= int(vsa.dim * SIM_EXPLAIN_THRESHOLD):
                    outcomes[i].append(outcome[i])
                    sim_to_orig[i].append(sim_orig)
                    # sim_to_remain[i].append(sim_remain)
                    inputs[i] = inputs[i] - vsa.expand(vector)
                    explained = "EXPLAINED"

                debug_message += f"DEBUG: outcome = {outcome[i]}, sim_orig = {round(sim_orig.item()/DIM, 3)}, sim_remain = {round(sim_remain.item()/DIM, 3)}, energy_left = {round(vsa.energy(inputs[i]).item()/DIM,3)}, {converge}, {explained}\n"

            # If the final t trial all generate low similarity, likely no more vectors to be extracted and stop
            t = 3
            if (k >= t-1):
                try:
                    if (all(torch.stack(sim_to_orig_all[0][-t:]) < int(vsa.dim * EARLY_TERM_THRESHOLD))):
                        break
                except:
                    pass

            # If energy left in the input is too low, likely no more vectors to be extracted and stop
            # When inputs are batched, must wait until all inputs are exhausted
            if (all(vsa.energy(inputs) <= int(vsa.dim * ENERGY_THRESHOLD))):
                break


        # Split batch results
        for i in range(len(inputs)):
            debug_message += f"DEBUG: pre-filtered: {outcomes[i]}\n"
            outcomes[i] = [outcomes[i][j] for j in range(len(outcomes[i])) if sim_to_orig[i][j] >= int(vsa.dim * SIM_DETECT_THRESHOLD)]

        counts = [len(outcomes[i]) for i in range(len(outcomes))]

        return outcomes, unconverged, iters, counts, debug_message

    rn = Resonator(vsa, mode=VSA_MODE, type=RESONATOR_TYPE, activation=ACTIVATION, act_val=ACT_VALUE, iterations=NUM_ITERATIONS, stoch=STOCHASTICITY, randomness=RANDOMNESS, early_converge=EARLY_CONVERGE, seed=SEED, device=device)

    init_estimates = rn.get_init_estimates().unsqueeze(0).repeat(TEST_BATCH_SIZE,1,1)

    incorrect_count = [0] * MAX_NUM_OBJECTS
    unconverged = [[0,0] for _ in range(MAX_NUM_OBJECTS)]    # [correct, incorrect]
    total_iters = [0] * MAX_NUM_OBJECTS
    n = 0
    # images in tensor([B, H, W, C])
    # labels in [{'pos_x': tensor, 'pos_y': tensor, 'color': tensor, 'digit': tensor}, ...]
    # targets in VSATensor([B, D])
    for images, labels, targets in tqdm(test_dl, desc="Test", leave=True if VERBOSE >= 1 else False):
        images = images.to(device)
        images_nchw = (images.type(torch.float32)/255).permute(0,3,1,2)
        infer_result = model(images_nchw)

        # round() will round numbers near 0 to 0, which is not ideal when there's one object, since the vector should be bipolar
        # But 0 is legitimate when there are multiple objects.
        infer_result = infer_result.round().type(torch.int8)

        # Factorization
        outcomes, convergences, iters, counts, debug_message = factorization(vsa, rn, infer_result, init_estimates)

        # Compare results
        # Batch: multiple samples
        for i in range(len(labels)):
            incorrect = False
            message = ""
            label = labels[i]
            outcome = outcomes[i]
            convergence = convergences[i]
            count = counts[i]
            iter = iters[i]
            sim_per_obj = []
            result = []

            # Convert to labels
            for o in outcome:
                result.append(
                    {
                        'pos_x': o[0],
                        'pos_y': o[1],
                        'color': o[2],
                        'digit': o[3]
                    }
                )

            total_iters[len(label)-1] += sum(iter)

            if (count != len(label)):
                incorrect = True
                message += Fore.RED + "Incorrect number of vectors detected, got {}, expected {}".format(count, len(label)) + Fore.RESET + "\n"
            else:
                message += f"Correct number of vectors detected: {count} \n"

            # Sample: multiple objects
            for j in range(len(label)):
                # Incorrect if one object is not detected 
                # For n objects, only check the first n results
                if (label[j] not in result):
                    incorrect = True
                    message += Fore.RED + "Object {} is not detected.".format(label[j]) + Fore.RESET + "\n"
                else:
                    message += "Object {} is correctly detected.".format(label[j]) + "\n"

                # Collect per-object similarity
                sim_per_obj.append(round(get_similarity(infer_result[i], vsa.lookup([label[j]]), False).item(), 3))

            if incorrect:
                unconverged[len(label)-1][1] += convergence
                incorrect_count[len(label)-1] += 1
                if (VERBOSE >= 1):
                    print(Fore.BLUE + f"Test {n} Failed" + Fore.RESET)
                    print("Inference result similarity = {:.3f}".format(get_similarity(infer_result[i], targets[i], False).item()))
                    print("Per-object similarity = {}".format(sim_per_obj))
                    print(f"Unconverged: {convergence}")
                    print(f"Iterations: {iter}")
                    print(message[:-1])
                    print("Result = {}".format(result))
                    print(debug_message)
            else:
                unconverged[len(label)-1][0] += convergence
                if (VERBOSE >= 2):
                    print(Fore.BLUE + f"Test {n} Passed" + Fore.RESET)
                    print("Inference result similarity = {:.3f}".format(get_similarity(infer_result[i], targets[i], False).item()))
                    print("Per-object similarity = {}".format(sim_per_obj))
                    print(f"Unconverged: {convergence}")
                    print(f"Iterations: {iter}")
                    print(message[:-1])
                    print(debug_message)
            n += 1

    return incorrect_count, unconverged, total_iters

# TODO outdated
def test_algo2(vsa, model, test_dl, device):
    pass
    # def factorization(vsa, resonator_network, inputs, init_estimates, codebooks, orig_indices = None):
    #     result_set = [[] for _ in range(inputs.size(0))]
    #     converg_set = [[] for _ in range(inputs.size(0))]

    #     inputs = inputs.clone()
    #     inputs = vsa.quantize(inputs)

    #     for k in range(MAX_NUM_OBJECTS):
    #         # Manually unbind ID 
    #         inputs_ = vsa.bind(inputs, vsa.id_codebook[k])

    #         # Run resonator network
    #         result, convergence = resonator_network(inputs_, init_estimates, codebooks, orig_indices)

    #         # Split batch results
    #         for i in range(len(result)):
    #             result_set[i].append(
    #                 {
    #                     'pos_x': result[i][0],
    #                     'pos_y': result[i][1],
    #                     'color': result[i][2],
    #                     'digit': result[i][3]
    #                 }
    #             )
    #             converg_set[i].append(convergence)
    #     return result_set, converg_set

    # rn = Resonator(vsa, type=RESONATOR_TYPE, activation=ACTIVATION, iterations=NUM_ITERATIONS, device=device)

    # # Remove the ID codebook since it is manually unbound
    # codebooks = vsa.codebooks[:-1]

    # codebooks, orig_indices = rn.reorder_codebooks(codebooks)
    # init_estimates = rn.get_init_estimates(codebooks, TEST_BATCH_SIZE)
    # if QUANTIZE:
    #     init_estimates = vsa.quantize(init_estimates)

    # incorrect_count = [0] * MAX_NUM_OBJECTS
    # unconverged = [[0,0] for _ in range(MAX_NUM_OBJECTS)]    # [correct, incorrect]
    # n = 0

    # # images in tensor([B, H, W, C])
    # # labels in [{'pos_x': tensor, 'pos_y': tensor, 'color': tensor, 'digit': tensor}, ...]
    # # targets in VSATensor([B, D])
    # for images, labels, targets in tqdm(test_dl, desc="Test", leave=True if VERBOSE >= 1 else False):

    #     # Inference
    #     images = images.to(device)
    #     images_nchw = (images.type(torch.float32)/255).permute(0,3,1,2)
    #     infer_result = model(images_nchw)
    #     if VSA_MODE == "SOFTWARE":
    #         infer_result = infer_result.round().type(torch.int8)
    #     else:
    #         infer_result = torch.sigmoid(infer_result).round().type(torch.int8)

    #     # Factorization
    #     outcomes, convergence = factorization(vsa, rn, infer_result, init_estimates, codebooks, orig_indices)

    #     # Compare results
    #     # Batch: multiple samples
    #     for i in range(len(labels)):
    #         incorrect = False
    #         message = ""
    #         label = labels[i]
    #         sim_per_obj = []

    #         # Must look up the entire label (instead of one object at a time) together to get the correct reordering done
    #         gt_objs = vsa.lookup(label, bundled=False)

    #         # Sample: multiple objects
    #         for j in range(len(label)):
    #             # Incorrect if one object is not detected 
    #             # For n objects, only check the first n results
    #             if (label[j] not in outcomes[i][0: len(label)]):
    #                 message += Fore.RED + "Object {} is not detected.".format(label[j]) + Fore.RESET + "\n"
    #                 incorrect = True
    #                 unconverged[len(label)-1][1] += 1 if convergence[i][j] == NUM_ITERATIONS-1 else 0
    #             else:
    #                 message += "Object {} is correctly detected.".format(label[j]) + "\n"
    #                 unconverged[len(label)-1][0] += 1 if convergence[i][j] == NUM_ITERATIONS-1 else 0
                
    #             sim_per_obj.append(round(get_similarity(infer_result[i], gt_objs[j], NORMALIZE).item(), 3))

    #         if incorrect:
    #             incorrect_count[len(label)-1] += 1 if incorrect else 0
    #             if (VERBOSE >= 1):
    #                 print(Fore.BLUE + f"Test {n} Failed:      Convergence = {convergence[i]}" + Fore.RESET)
    #                 print("Inference result similarity = {:.3f}".format(get_similarity(infer_result[i], targets[i], NORMALIZE).item()))
    #                 print("Per-object similarity = {}".format(sim_per_obj))
    #                 print(message[:-1])
    #                 print("Outcome = {}".format(outcomes[i][0: len(label)]))
    #         else:
    #             if (VERBOSE >= 2):
    #                 print(Fore.BLUE + f"Test {n} Passed:      Convergence = {convergence[i]}" + Fore.RESET)
    #                 print("Inference result similarity = {:.3f}".format(get_similarity(infer_result[i], targets[i], NORMALIZE).item()))
    #                 print("Per-object similarity = {}".format(sim_per_obj))
    #                 print(message[:-1])
    #         n += 1

    # return incorrect_count, unconverged
        

if __name__ == "__main__":

    action = sys.argv[1] # train, test, datagen

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # if action == "test":
        # device = "cpu"
    print(f"Workload Config: algorithm = {ALGO}, vsa mode = {VSA_MODE}, dim = {DIM}, num pos x = {NUM_POS_X}, num pos y = {NUM_POS_Y}, num color = {NUM_COLOR}, num digits = 10, max num objects = {MAX_NUM_OBJECTS}")

    vsa = get_vsa(device)
    model = MultiConceptNonDecomposed(dim=DIM, device=device)

    if action == "train":
        print(f"Training on {device}: samples = {NUM_TRAIN_SAMPLES}, epochs = {TRAIN_EPOCH}, batch size = {TRAIN_BATCH_SIZE}")
        loss_fn = torch.nn.MSELoss() if ALGO == "algo1" else torch.nn.BCEWithLogitsLoss()

        if sys.argv[-1].endswith(".pt"):
            if os.path.exists(sys.argv[-1]):
                checkpoint = torch.load(sys.argv[-1])
                model.load_state_dict(checkpoint)
                print(f"On top of checkpoint {sys.argv[-1]}")
            else:
                print("Invalid model checkpoint path.")
                exit(1)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        cur_time_pst = datetime.now().astimezone(timezone('US/Pacific')).strftime("%m-%d-%H-%M")
        train_dl = get_train_data(vsa)
        final_loss = train(train_dl, model, loss_fn, optimizer, num_epoch=TRAIN_EPOCH, cur_time=cur_time_pst, device=device)
        model_weight_loc = os.path.join(test_dir, f"model_weights_{MAX_NUM_OBJECTS}objs{'_single_count' if SINGLE_COUNT else ''}_{TRAIN_BATCH_SIZE}batch_{TRAIN_EPOCH}epoch_{NUM_TRAIN_SAMPLES}samples_{final_loss}loss_{cur_time_pst}.pt")
        torch.save(model.state_dict(), model_weight_loc)
        print(f"Model weights saved to {model_weight_loc}")

    # Test mode
    elif action == "test":
        print(f"Running test on {device}, batch size = {TEST_BATCH_SIZE}")

        # assume we provided checkpoint path at the end of the command line
        if sys.argv[-1].endswith(".pt") and os.path.exists(sys.argv[-1]):
            checkpoint = torch.load(sys.argv[-1])
            model.load_state_dict(checkpoint)
        else:
            print("Please provide a valid model checkpoint path.")
            exit(1)

        model.eval()

        print(Fore.CYAN + f"""
Resonator setup:  max_trials = {MAX_TRIALS}, energy_thresh = {ENERGY_THRESHOLD}, similarity_explain_thresh = {SIM_EXPLAIN_THRESHOLD}, \
similarity_detect_thresh = {SIM_DETECT_THRESHOLD}, expanded_hd_bits = {EHD_BITS}, int_reg_bits = {SIM_BITS}, 
resonator = {RESONATOR_TYPE}, iterations = {NUM_ITERATIONS}, stochasticity = {STOCHASTICITY}, randomness = {RANDOMNESS}, \
activation = {ACTIVATION}, act_val = {ACT_VALUE}, early_converge_thresh = {EARLY_CONVERGE}
""" + Fore.RESET)

        test_dl = get_test_data(vsa)
 
        if ALGO == "algo1":
            incorrect_count, unconverged, total_iters = test_algo1(vsa, model, test_dl, device)
        elif ALGO == "algo2":
            incorrect_count, unconverged, total_iters = test_algo2(vsa, model, test_dl, device)

        if SINGLE_COUNT:
            print(f"{MAX_NUM_OBJECTS} objects: Accuracy = {NUM_TEST_SAMPLES - incorrect_count[MAX_NUM_OBJECTS-1]}/{NUM_TEST_SAMPLES}   Unconverged = {unconverged[MAX_NUM_OBJECTS-1]}    Average iterations: {total_iters[MAX_NUM_OBJECTS-1] / (NUM_TEST_SAMPLES)}")
        else:
            for i in range(MAX_NUM_OBJECTS):
                print(f"{i+1} objects: Accuracy = {NUM_TEST_SAMPLES//MAX_NUM_OBJECTS - incorrect_count[i]}/{NUM_TEST_SAMPLES//MAX_NUM_OBJECTS}   Unconverged = {unconverged[i]}    Average iterations: {total_iters[i] / (NUM_TEST_SAMPLES//MAX_NUM_OBJECTS)}")

    # Data Gen mode      
    elif action == "datagen":
        MultiConceptMNIST(test_dir, vsa, train=True, num_samples=NUM_TRAIN_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, single_count=SINGLE_COUNT, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR, force_gen=True)
        MultiConceptMNIST(test_dir, vsa, train=False, num_samples=NUM_TEST_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, single_count=SINGLE_COUNT, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR, force_gen=True)

    

