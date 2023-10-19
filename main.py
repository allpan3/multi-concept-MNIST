
import torch
from models.vsa import MultiConceptMNISTVSA1, MultiConceptMNISTVSA2
from vsa import Resonator, VSA
from datasets.dataset import MultiConceptMNIST
from datasets import *
import torch
from tqdm import tqdm
from models.nn_non_decomposed import MultiConceptNonDecomposed
from colorama import Fore
from torch.utils.tensorboard import SummaryWriter
import sys
import os
from datetime import datetime
from pytz import timezone
from tools.model_quantization import quantize_model
import torchvision.transforms as transforms

###########
# Configs #
###########
VERBOSE = 2
SEED = 0
ALGO = "algo1" # "algo1", "algo2"
VSA_MODE = "HARDWARE" # "SOFTWARE", "HARDWARE"
QUANTIZE_MODEL = False 
DIM = 1024
MAX_NUM_OBJECTS = 3
SINGLE_COUNT = False # True, False
NUM_POS_X = 3
NUM_POS_Y = 3
NUM_COLOR = 7
# Hardware config
FOLD_DIM = 256
EHD_BITS = 9
SIM_BITS = 13
# Train
TRAIN_EPOCH = 50
TRAIN_BATCH_SIZE = 128
NUM_TRAIN_SAMPLES = 10000
# Test
TEST_BATCH_SIZE = 1
NUM_TEST_SAMPLES = 300
# Resonator
RESONATOR_TYPE = "SEQUENTIAL" # "SEQUENTIAL", "CONCURRENT"
MAX_TRIALS = MAX_NUM_OBJECTS + 10
NUM_ITERATIONS = 200
ACTIVATION = 'THRESH_AND_SCALE'      # 'IDENTITY', 'THRESHOLD', 'SCALEDOWN', "THRESH_AND_SCALE"
ACT_VALUE = 16
STOCHASTICITY = "SIMILARITY"  # apply stochasticity: "NONE", "SIMILARITY", "VECTOR"
RANDOMNESS = 0.04
# Similarity thresholds are affected by the maximum number of vectors superposed. These values need to be lowered when more vectors are superposed
SIM_EXPLAIN_THRESHOLD = 0.25
SIM_DETECT_THRESHOLD = 0.12
ENERGY_THRESHOLD = 0.25
EARLY_CONVERGE = None
EARLY_TERM_THRESHOLD = 0.2     # Compared to remaining

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


test_dir = f"./tests/{VSA_MODE}-{DIM}dim{'-' + str(FOLD_DIM) + 'fd' if VSA_MODE=='HARDWARE' else ''}-{NUM_POS_X}x-{NUM_POS_Y}y-{NUM_COLOR}color/{ALGO}"


def train(dataloader, model, loss_fn, optimizer, num_epoch, cur_time, device = "cpu"):
    writer = SummaryWriter(log_dir=f"./runs/{cur_time}-{ALGO}-{VSA_MODE}-{DIM}dim{'-' + str(FOLD_DIM) + 'fd' if VSA_MODE=='HARDWARE' else ''}-{MAX_NUM_OBJECTS}objs-{NUM_POS_X}x-{NUM_POS_Y}y-{NUM_COLOR}color", filename_suffix=f".{TRAIN_BATCH_SIZE}batch-{TRAIN_EPOCH}epoch-{NUM_TRAIN_SAMPLES}samples")
    # images in tensor([B, C, H, W])
    # labels in [(pos_x, pos_y, color, digit), ...]
    # targets in VSATensor([B, D])
    for epoch in range(num_epoch):
        if epoch != 0 and epoch % 5 == 0:
            model_weight_loc = os.path.join(test_dir, f"model_weights_{MAX_NUM_OBJECTS}objs{'_single_count' if SINGLE_COUNT else ''}_{TRAIN_BATCH_SIZE}batch_{epoch}epoch_{NUM_TRAIN_SAMPLES}samples_{round(loss.item(),4)}loss_{cur_time}.pt")
            torch.save(model.state_dict(), model_weight_loc)
            print(f"Model checkpoint saved to {model_weight_loc}")

        for idx, (images, labels, targets) in enumerate(tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)):
            # Converts to [0, 1]
            images = get_transform()(images.to(device))
            targets_float = targets.to(device).type(torch.float32)
            output = model(images)
            loss = loss_fn(output, targets_float)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
            writer.add_scalar('Loss/train', loss, epoch * len(dataloader) + idx)
        # import pdb; pdb.set_trace()

    return round(loss.item(), 4)

def get_dot_similarity(v1, v2, quantized):
    """
    Return the normalized dot similarity for quantized vectors. Quantize if not quantized
    """
    if not quantized:
        return VSA.dot_similarity(VSA.quantize(v1), VSA.quantize(v2)) / DIM
    else:
        return VSA.dot_similarity(v1, v2) / DIM

def get_cos_similarity(input, others):
    """
    Return the cosine similarity.
    """
    input_dot = torch.sum(input * input, dim=-1)
    input_mag = torch.sqrt(input_dot)
    others_dot = torch.sum(others * others, dim=-1)
    others_mag = torch.sqrt(others_dot)
    if input.dim() >= 2:
        magnitude = input_mag.unsqueeze(-1) * others_mag.unsqueeze(-2)
    else:
        magnitude = input_mag * others_mag
    magnitude = torch.clamp(magnitude, min=1e-08)

    if others.dim() >= 2:
        others = others.transpose(-2, -1)

    return torch.matmul(input.type(torch.float32), others.type(torch.float32)) / magnitude
            
def get_vsa(device):
    if ALGO == "algo1":
        vsa = MultiConceptMNISTVSA1(test_dir, mode=VSA_MODE, dim=DIM, max_num_objects=MAX_NUM_OBJECTS, num_colors=NUM_COLOR, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, fold_dim=FOLD_DIM, ehd_bits=EHD_BITS, sim_bits=SIM_BITS, seed=SEED, device=device)
    elif ALGO == "algo2":
        vsa = MultiConceptMNISTVSA2(test_dir, mode=VSA_MODE, dim=DIM, max_num_objects=MAX_NUM_OBJECTS, num_colors=NUM_COLOR, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, fold_dim=FOLD_DIM, ehd_bits=EHD_BITS, sim_bits=SIM_BITS, seed=SEED, device=device)
    return vsa

def get_transform():
    """
    Resize image to 224x224
    Typically we'd transform hte image when loading the dataset, but it consumes a too much memory, so we call this before inference
    """
    if QUANTIZE_MODEL:
        return transforms.Compose([
            # transforms.Resize(224, antialias=True)
        ])
    else:
        return transforms.Compose([
            # transforms.Resize(224, antialias=True),
            transforms.ConvertImageDtype(torch.float32)  # Converts to [0, 1]
        ])

def factorization_algo1(vsa, rn, inputs, init_estimates, codebooks=None, known=None):

    inputs = inputs.clone()
    init_estimates = init_estimates.clone()

    # if unbatched, add a batch dimension of 1 for easier processing
    unbatched = False
    if (inputs.dim() == 1):
        unbatched = True
        inputs = inputs.unsqueeze(0)
        init_estimates = init_estimates.unsqueeze(0)
    
    outcomes = [[] for _ in range(inputs.size(0))]  # num of batches
    unconverged = [0] * inputs.size(0)
    iters = [[] for _ in range(inputs.size(0))]
    sim_to_remain = [[] for _ in range(inputs.size(0))]
    sim_to_remain_all = [[] for _ in range(inputs.size(0))]
    sim_to_orig = [[] for _ in range(inputs.size(0))]
    sim_to_orig_all = [[] for _ in range(inputs.size(0))]
    debug_message = ""

    inputs_q = vsa.quantize(inputs)

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
        outcome, iter, converge = rn(inputs_, init_estimates, codebooks=codebooks, known=known)

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

            debug_message += f"DEBUG: outcome = {outcome[i]}, sim_orig = {round(sim_orig.item()/DIM, 3)}, sim_remain = {round(sim_remain.item()/DIM, 3)}, energy_left = {round(vsa.energy(inputs[i]).item()/DIM,3)}, iter = {iter}, {converge}, {explained}\n"

        # The early terminate part can only be supported with batch size = 1
        assert TEST_BATCH_SIZE == 1, "Batch size != 1 is not yet supported"
        # If the final t trial all generate low similarity, likely no more vectors to be extracted and stop
        t = 3
        if (k >= t-1):
            try:
                if (all(torch.stack(sim_to_remain_all[0][-t:]) < int(vsa.dim * EARLY_TERM_THRESHOLD))):
                    break
            except:
                pass

        # If energy left in the input is too low, likely no more vectors to be extracted and stop
        # When inputs are batched, must wait until all inputs are exhausted
        if (all(vsa.energy(inputs) <= int(vsa.dim * ENERGY_THRESHOLD))):
            break

    # Filter output based on similarity threshold
    for i in range(len(inputs)):
        debug_message += f"DEBUG: pre-filtered: {outcomes[i]}\n"
        outcomes[i] = [outcomes[i][j] for j in range(len(outcomes[i])) if sim_to_orig[i][j] >= int(vsa.dim * SIM_DETECT_THRESHOLD)]
        # Since we know there'll be no overlapped objects in this workload, we can filter out duplicates
        # Duplicates can appear if one of the objects is more similar to the input than the other (due to biased noise of NN)
        # outcomes[i] = list(set(outcomes[i]))
    
    counts = [len(outcomes[i]) for i in range(len(outcomes))]

    # if unbatched, remove the batch dimension to be consistent with caller
    if unbatched:
        outcomes = outcomes[0]
        unconverged = unconverged[0]
        iters = iters[0]
        counts = counts[0]

    return outcomes, unconverged, iters, counts, debug_message


def test_algo1(vsa, model, test_dl, device):
    """
    Algorithm 1: explain away
    See VSA-factorization repo for details
    NN needs to be trained with unquantized vectors
    """

    rn = Resonator(vsa, mode=VSA_MODE, type=RESONATOR_TYPE, activation=ACTIVATION, act_val=ACT_VALUE, iterations=NUM_ITERATIONS, stoch=STOCHASTICITY, randomness=RANDOMNESS, early_converge=EARLY_CONVERGE, seed=SEED, device=device)

    init_estimates = rn.get_init_estimates().unsqueeze(0).repeat(TEST_BATCH_SIZE,1,1)

    incorrect_count = [0] * MAX_NUM_OBJECTS
    unconverged = [[0,0] for _ in range(MAX_NUM_OBJECTS)]    # [correct, incorrect]
    total_iters = [0] * MAX_NUM_OBJECTS
    n = 0
    for images, labels, targets, _ in tqdm(test_dl, desc="Test", leave=True if VERBOSE >= 1 else False):
        images = get_transform()(images.to(device))
        infer_result = model(images)

        if QUANTIZE_MODEL:
            # infer_result = (infer_result/ 128.0 * MAX_NUM_OBJECTS).round().type(torch.int8)
            pass
        else:
            # round() will round numbers near 0 to 0, which is not ideal when there's one object, since the vector should be bipolar (either 1 or -1)
            # But 0 is legitimate when there are even number of multiple objects.
            infer_result = infer_result.round().type(torch.int8)

        # Factorization
        outcomes, convergences, iters, counts, debug_message = factorization_algo1(vsa, rn, infer_result, init_estimates)

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

            total_iters[len(label)-1] += sum(iter)

            if (count != len(label)):
                incorrect = True
                message += Fore.RED + "Incorrect number of vectors detected, got {}, expected {}".format(count, len(label)) + Fore.RESET + "\n"
            else:
                message += f"Correct number of vectors detected: {count} \n"

            # Sample: multiple objects
            for j in range(len(label)):
                # For per-object similarity, compare the quantized vectors, which are what the resonator network sees
                sim_per_obj = round(get_dot_similarity(infer_result[i], vsa.lookup([label[j]]), False).item(), 3) # Adding the [] makes it unquantized
                # Incorrect if one object is not detected 
                # For n objects, only check the first n results
                if (label[j] not in outcome):
                    incorrect = True
                    message += Fore.RED + "Object {} is not detected. Similarity = {}".format(label[j], sim_per_obj) + Fore.RESET + "\n"
                else:
                    message += "Object {} is correctly detected. Similarity = {}".format(label[j], sim_per_obj) + "\n"

            if incorrect:
                unconverged[len(label)-1][1] += convergence
                incorrect_count[len(label)-1] += 1
                if (VERBOSE >= 1):
                    print(Fore.BLUE + f"Test {n} Failed" + Fore.RESET)
                    print("Inference result similarity = {:.3f}".format(get_cos_similarity(infer_result[i], targets[i]).item()))
                    # print("Per-object similarity = {}".format(sim_per_obj))
                    print(f"Unconverged: {convergence}")
                    print(f"Iterations: {iter}")
                    print(message[:-1])
                    print(debug_message[:-1])
                    print("Outcome: {}".format(outcome))
            else:
                unconverged[len(label)-1][0] += convergence
                if (VERBOSE >= 2):
                    print(Fore.BLUE + f"Test {n} Passed" + Fore.RESET)
                    print("Inference result similarity = {:.3f}".format(get_cos_similarity(infer_result[i], targets[i]).item()))
                    print(f"Unconverged: {convergence}")
                    print(f"Iterations: {iter}")
                    print(message[:-1])
                    print(debug_message[:-1])
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

    # for images, labels, targets in tqdm(test_dl, desc="Test", leave=True if VERBOSE >= 1 else False):

    #     # Inference
    #     images = images.to(device)
    # # images_nchw = (images.type(torch.float32)/255)
    #     infer_result = model(images)
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


def reason_algo1(vsa, model, test_dl, device):
    rn = Resonator(vsa, mode=VSA_MODE, type=RESONATOR_TYPE, activation=ACTIVATION, act_val=ACT_VALUE, iterations=NUM_ITERATIONS, stoch=STOCHASTICITY, randomness=RANDOMNESS, early_converge=EARLY_CONVERGE, seed=SEED, device=device)

    init_estimates = rn.get_init_estimates().unsqueeze(0).repeat(TEST_BATCH_SIZE,1,1)

    incorrect_count = [0] * MAX_NUM_OBJECTS
    # unconverged = [[0,0] for _ in range(MAX_NUM_OBJECTS)]    # [correct, incorrect]
    # total_iters = [0] * MAX_NUM_OBJECTS
    n = 0
    for images, labels, targets, questions in tqdm(test_dl, desc="Reason", leave=True if VERBOSE >= 1 else False):
        images = get_transform()(images.to(device))
        infer_result = model(images)
        infer_result = infer_result.round().type(torch.int8) 
        # infer_result = targets

        # Only supports batch size = 1
        label = labels[0]
        questions = questions[0]
        input_q = vsa.quantize(infer_result[0])

        for x in range(len(questions)):
            q = questions[x]
            message = ""
            query = tuple(q["query"])
            if q["question"] == "object_exists":
                query_vector = vsa.get_vector(query)
                if all(query[i] != None for i in range(len(query))):
                    # If the question specifies all attributes, directly compare the inference result to the ground truth
                    sim = vsa.dot_similarity(input_q, query_vector)
                else:
                    # Get the codebooks of unknown factors
                    codebooks = [vsa.codebooks[i] for i in range(len(query)) if query[i] == None]
                    # Factorize the remaining vector
                    init_estimates = rn.get_init_estimates(codebooks)
                    outcome, iter, converge = rn(input_q, init_estimates, codebooks=codebooks, known=query)
                    message += f"Convergence: {converge}\n"

                    # Compare the resultant compositional vector to the original input vector
                    # * Technically we should be able to just look at the confidence of extracted unkown factors and determine if they are correct
                    # * but the issue is that the optimal configuration for resonator network varies depending on the number of codebooks and their codevectors, 
                    # * and we don't know in run time what configuration to use for the particular question (e.g. stochasticity, activation, etc.) 
                    # * Aggressive configurations will always lead to convergence and the results may be incorrect. Vanilla configuration (no stochasticity, no activation) may
                    # * be unable to factorize valid vectors.
                    # * So our solution is to re-compose the vector using extracted factors and compare it to the original input vector.
                    sim = vsa.dot_similarity(input_q, vsa.get_vector(outcome))

                message += "Iter = {}\n".format(iter)
                message += "Resultant outcome = {}\n".format(outcome) 
                message += "Resultant similarity = {:.3f}\n".format(sim.item()) 

                if (sim > vsa.dim * 0.15):
                    answer = True
                else:
                    answer = False

            elif q["question"] == "object_count": 
                # Get the codebooks of unknown factors
                codebooks = [vsa.codebooks[i] for i in range(len(query)) if query[i] == None]
                # Factorize the remaining vector
                init_estimates = rn.get_init_estimates(codebooks)
                # Have to pass unquantized inputs to the factorizer to extract multiple objects
                outcome, convergence, iters, count, debug_message = factorization_algo1(vsa, rn, infer_result[0], init_estimates, codebooks=codebooks, known=query)

                answer = count

                message += f"Outcome = {outcome}\n"
                message += f"Convergence = {convergence}\n"
                message += debug_message
            else:
                answer = q["answer"]
                pass

            sim_per_obj = []
            # Collect per-object similarity
            for j in range(len(label)):
                sim_per_obj.append(round(get_dot_similarity(infer_result[0], vsa.lookup([label[j]]), False).item(), 3))

            if (answer != q["answer"]):
                incorrect_count[len(label)-1] += 1
                if VERBOSE >= 1:
                    print(Fore.RED + f"Test {n} Question {x} Wrong" + Fore.RESET)
                    print("Prompt:", q["prompt"])
                    print("Inference result similarity = {:.3f}".format(get_cos_similarity(infer_result[0], targets[0]).item()))
                    print("Per-object similarity = {}".format(sim_per_obj))
                    print("Label: {}".format(label))
                    print("Query: {}".format(query))
                    print("Answer: {}".format(q["answer"]))
                    print("Got   : {}".format(answer))
                    print(message[:-1])
            else:
                if VERBOSE >= 2:
                    print(Fore.BLUE + f"Test {n} Question {x} Correct" + Fore.RESET)
                    print("Prompt:", q["prompt"])
                    print("Inference result similarity = {:.3f}".format(get_cos_similarity(infer_result[0], targets[0]).item()))
                    print("Per-object similarity = {}".format(sim_per_obj))
                    print("Label: {}".format(label))
                    print("Query: {}".format(query))
                    print("Answer: {}".format(q["answer"]))
                    print(message[:-1])
        n += 1

    return incorrect_count

if __name__ == "__main__":

    action = sys.argv[1] # train, test, datagen, eval, reason

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # if action == "test":
    #     device = "cpu"
    print(f"Workload Config: algorithm = {ALGO}, vsa mode = {VSA_MODE}, dim = {DIM}, num pos x = {NUM_POS_X}, num pos y = {NUM_POS_Y}, num color = {NUM_COLOR}, num digits = 10, max num objects = {MAX_NUM_OBJECTS}")

    vsa = get_vsa(device)
    model = MultiConceptNonDecomposed(dim=DIM, device=device)

    if action == "train":
        print(f"Training on {device}: samples = {NUM_TRAIN_SAMPLES}, epochs = {TRAIN_EPOCH}, batch size = {TRAIN_BATCH_SIZE}")

        if sys.argv[-1].endswith(".pt"):
            if os.path.exists(sys.argv[-1]):
                checkpoint = torch.load(sys.argv[-1], map_location=device)
                model.load_state_dict(checkpoint)
                print(f"On top of checkpoint {sys.argv[-1]}")
            else:
                print("Invalid model checkpoint path.")
                exit(1)

        loss_fn = torch.nn.MSELoss() if ALGO == "algo1" else torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        cur_time_pst = datetime.now().astimezone(timezone('US/Pacific')).strftime("%m-%d-%H-%M")
        train_dl = get_train_data(test_dir, vsa, NUM_TRAIN_SAMPLES, MAX_NUM_OBJECTS, SINGLE_COUNT, TRAIN_BATCH_SIZE, NUM_POS_X, NUM_POS_Y, NUM_COLOR)
        final_loss = train(train_dl, model, loss_fn, optimizer, num_epoch=TRAIN_EPOCH, cur_time=cur_time_pst, device=device)
        model_weight_loc = os.path.join(test_dir, f"model_weights_{MAX_NUM_OBJECTS}objs{'_single_count' if SINGLE_COUNT else ''}_{TRAIN_BATCH_SIZE}batch_{TRAIN_EPOCH}epoch_{NUM_TRAIN_SAMPLES}samples_{final_loss}loss_{cur_time_pst}.pt")
        torch.save(model.state_dict(), model_weight_loc)
        print(f"Model weights saved to {model_weight_loc}")

    # Test mode
    elif action == "test":
        print(f"Running test on {device}, batch size = {TEST_BATCH_SIZE}")

        # assume we provided checkpoint path at the end of the command line
        if sys.argv[-1].endswith(".pt") and os.path.exists(sys.argv[-1]):
            checkpoint = torch.load(sys.argv[-1], map_location=device)
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

        test_dl = get_test_data(test_dir, vsa, False, NUM_TEST_SAMPLES, MAX_NUM_OBJECTS, SINGLE_COUNT, TEST_BATCH_SIZE, NUM_POS_X, NUM_POS_Y, NUM_COLOR)
        quan_dl = get_test_data(test_dir, vsa, True, NUM_TEST_SAMPLES if NUM_TEST_SAMPLES < 300 else 300, MAX_NUM_OBJECTS, SINGLE_COUNT, TEST_BATCH_SIZE, NUM_POS_X, NUM_POS_Y, NUM_COLOR)

        if QUANTIZE_MODEL:
            quantize_model(model, quan_dl)

        if ALGO == "algo1":
            incorrect_count, unconverged, total_iters = test_algo1(vsa, model, test_dl, device)
        elif ALGO == "algo2":
            incorrect_count, unconverged, total_iters = test_algo2(vsa, model, test_dl, device)

        if SINGLE_COUNT:
            print(f"{MAX_NUM_OBJECTS} objects: Accuracy = {NUM_TEST_SAMPLES - incorrect_count[MAX_NUM_OBJECTS-1]}/{NUM_TEST_SAMPLES}   Unconverged = {unconverged[MAX_NUM_OBJECTS-1]}    Average iterations: {total_iters[MAX_NUM_OBJECTS-1] / (NUM_TEST_SAMPLES)}")
        else:
            for i in range(MAX_NUM_OBJECTS):
                print(f"{i+1} objects: Accuracy = {NUM_TEST_SAMPLES//MAX_NUM_OBJECTS - incorrect_count[i]}/{NUM_TEST_SAMPLES//MAX_NUM_OBJECTS}   Unconverged = {unconverged[i]}    Average iterations: {total_iters[i] / (NUM_TEST_SAMPLES//MAX_NUM_OBJECTS)}")

    elif action == "eval":
        assert TEST_BATCH_SIZE == 1, "Evaluation mode only supports batch size = 1 so far"
        # assume we provided checkpoint path at the end of the command line
        if sys.argv[-1].endswith(".pt") and os.path.exists(sys.argv[-1]):
            checkpoint = torch.load(sys.argv[-1], map_location=device)
            model.load_state_dict(checkpoint)
        else:
            print("Please provide a valid model checkpoint path.")
            exit(1)

        model.eval()

        test_dl = get_test_data(test_dir, vsa, False, NUM_TEST_SAMPLES, MAX_NUM_OBJECTS, SINGLE_COUNT, TEST_BATCH_SIZE, NUM_POS_X, NUM_POS_Y, NUM_COLOR)

        if QUANTIZE_MODEL:
            quan_dl = get_test_data(test_dir, vsa, True, NUM_TEST_SAMPLES if NUM_TEST_SAMPLES < 300 else 300, MAX_NUM_OBJECTS, SINGLE_COUNT, TEST_BATCH_SIZE, NUM_POS_X, NUM_POS_Y, NUM_COLOR)
            quantize_model(model, quan_dl)

        total_sim = [0] * MAX_NUM_OBJECTS
        max_sim = [-DIM] * MAX_NUM_OBJECTS
        min_sim = [DIM] * MAX_NUM_OBJECTS

        for images, labels, targets, _ in tqdm(test_dl, desc="Eval", leave=True if VERBOSE >= 1 else False):
            images = get_transform()(images.to(device))
            infer_result = model(images)

            if QUANTIZE_MODEL:
                # infer_result = (infer_result/ 128.0 * MAX_NUM_OBJECTS).round().type(torch.int8)
                pass
            else:
                # round() will round numbers near 0 to 0, which is not ideal when there's one object, since the vector should be bipolar (either 1 or -1)
                # But 0 is legitimate when there are even number of multiple objects.
                infer_result = infer_result.round().type(torch.int8)

            vector_quantized = False if ALGO == "algo1" else True
            sim = torch.sum(get_cos_similarity(infer_result, targets)).item()
            # # Makes more sense to look at the average of per-object similarities, which ultimately determines whether all objects can be extracted
            # per_test_sim = 0
            # for label in labels[0]:
            #     per_test_sim += get_dot_similarity(infer_result, vsa.lookup([label]), vector_quantized).item()
            
            # sim = per_test_sim / len(labels[0])

            total_sim[len(labels[0])-1] += sim

            if (sim > max_sim[len(labels[0])-1]):
                max_sim[len(labels[0])-1] = sim
            if (sim < min_sim[len(labels[0])-1]):
                min_sim[len(labels[0])-1] = sim

        if SINGLE_COUNT:
            print(f"{MAX_NUM_OBJECTS} objects: Average similarity = {round(total_sim[MAX_NUM_OBJECTS-1] / NUM_TEST_SAMPLES, 3)}   Max similarity = {round(max_sim[MAX_NUM_OBJECTS-1], 3)}    Min similarity = {round(min_sim[MAX_NUM_OBJECTS-1], 3)}")
        else:
            for i in range(MAX_NUM_OBJECTS):
                print(f"{i+1} objects: Average similarity = {round(total_sim[i] / (NUM_TEST_SAMPLES//MAX_NUM_OBJECTS), 3)}   Max similarity = {round(max_sim[i], 3)}    Min similarity = {round(min_sim[i], 3)}")
            
            print(f"Avrage similarity = {round(sum(total_sim) / NUM_TEST_SAMPLES, 3)}")
 
    elif action == "reason":
        assert TEST_BATCH_SIZE == 1, "Reason mode only supports batch size = 1 so far"
        # assume we provided checkpoint path at the end of the command line
        if sys.argv[-1].endswith(".pt") and os.path.exists(sys.argv[-1]):
            checkpoint = torch.load(sys.argv[-1], map_location=device)
            model.load_state_dict(checkpoint)
        else:
            print("Please provide a valid model checkpoint path.")
            exit(1)

        model.eval()
        test_dl = get_test_data(test_dir, vsa, False, NUM_TEST_SAMPLES, MAX_NUM_OBJECTS, SINGLE_COUNT, TEST_BATCH_SIZE, NUM_POS_X, NUM_POS_Y, NUM_COLOR)
 
        if ALGO == "algo1":
            incorrect_count = reason_algo1(vsa, model, test_dl, device)

        if SINGLE_COUNT:
            print(f"{MAX_NUM_OBJECTS} objects: Accuracy = {NUM_TEST_SAMPLES - incorrect_count[MAX_NUM_OBJECTS-1]}/{NUM_TEST_SAMPLES}")
        else:
            for i in range(MAX_NUM_OBJECTS):
                print(f"{i+1} objects: Accuracy = {NUM_TEST_SAMPLES//MAX_NUM_OBJECTS - incorrect_count[i]}/{NUM_TEST_SAMPLES//MAX_NUM_OBJECTS}")

    # Data Gen mode      
    elif action == "datagen":
        MultiConceptMNIST(test_dir, vsa, train=True, num_samples=NUM_TRAIN_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, single_count=SINGLE_COUNT, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR, force_gen=True)
        MultiConceptMNIST(test_dir, vsa, train=False, num_samples=NUM_TEST_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, single_count=SINGLE_COUNT, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR, force_gen=True)
