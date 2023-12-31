
import torch
from models.vsa import get_vsa
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
from config import *
import argparse

test_dir = f"./tests/{VSA_MODE}-{DIM}dim{'-' + str(FOLD_DIM) + 'fd' if VSA_MODE=='HARDWARE' else ''}-{NUM_POS_X}x-{NUM_POS_Y}y-{NUM_COLOR}color/{ALGO}"

def train(dataloader, model, loss_fn, optimizer, num_epoch, cur_time, device = "cpu"):
    writer = SummaryWriter(log_dir=f"./runs/{cur_time}-{ALGO}-{VSA_MODE}-{DIM}dim{'-' + str(FOLD_DIM) + 'fd' if VSA_MODE=='HARDWARE' else ''}-{MAX_NUM_OBJECTS}objs-{NUM_POS_X}x-{NUM_POS_Y}y-{NUM_COLOR}color", filename_suffix=f".{TRAIN_BATCH_SIZE}batch-{TRAIN_EPOCH}epoch-{NUM_TRAIN_SAMPLES}samples")
    # images in tensor([B, C, H, W])
    # labels in [(pos_x, pos_y, color, digit), ...]
    # targets in VSATensor([B, D])
    for epoch in range(num_epoch):
        if epoch != 0 and epoch % 5 == 0:
            model_weight_loc = os.path.join(test_dir, f"model_weights_{MAX_NUM_OBJECTS}objs{'_single_count' if SINGLE_COUNT else ''}_{TRAIN_BATCH_SIZE}batch_{NUM_TRAIN_SAMPLES}samples_{epoch}epoch_{round(loss.item(),4)}loss_{cur_time}.pt")
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
    others_dot = torch.sum(others * others, dim=-1).to(input.device)
    others_mag = torch.sqrt(others_dot)
    if input.dim() >= 2:
        magnitude = input_mag.unsqueeze(-1) * others_mag.unsqueeze(-2)
    else:
        magnitude = input_mag * others_mag
    magnitude = torch.clamp(magnitude, min=1e-08)

    if others.dim() >= 2:
        others = others.transpose(-2, -1)

    return torch.matmul(input.type(torch.float32), others.type(torch.float32).to(input.device)) / magnitude


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

def factorization_algo1(vsa, rn, inputs, init_estimates, count=None, codebooks=None, known_factor=None):

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
        # So far performance is good even without this. Haven't seen too big of a difference.
        # if vsa.mode == "HARDWARE":
        #     # This is one way to randomize the initial estimates
        #     init_estimates = vsa.ca90(init_estimates)
        # elif vsa.mode == "SOFTWARE":
        #     # Replace this with true random vector
        #     init_estimates = vsa.apply_noise(init_estimates, 0.5)

        # Run resonator network
        outcome, iter, converge = rn(inputs_, init_estimates, codebooks=codebooks, known=known_factor)

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

        # If the final t trial all generate low similarity, likely no more vectors to be extracted and stop
        t = 3
        if (k >= t-1):
            try:
                if ((torch.Tensor(sim_to_remain_all[:][-t:]) < int(vsa.dim * EARLY_TERM_THRESHOLD)).all()):
                    break
            except:
                pass

        # If energy left in the input is too low, likely no more vectors to be extracted and stop
        # When inputs are batched, must wait until all inputs are exhausted
        if ((vsa.energy(inputs) <= int(vsa.dim * ENERGY_THRESHOLD)).all()):
            break

        # This condition never triggers when the explain away threshold is high enough. Need further test to see whether it causes issue
        # # When count is known, we should be able to break out of the loop slightly early
        # # But allow some extra objects so that we can rank the results by similarity
        # if count is not None:
        #     if all([len(outcomes[i]) >= count + 2 for i in range(len(outcomes))]):
        #         break

    # When the count is known, rank the results by similarity and pick the top k
    if count is not None:
        # Among all qualified outcomes, select the n closest to the original input
        # Split batch results
        for i in range(len(inputs)):
            debug_message += f"DEBUG: pre-ranked: {outcomes[i]}\n"
            # It's possible that none of the vectors extracted are similar enough to be considered as condidates
            if len(outcomes[i]) != 0:
                # Ranking by similarity to the original inputs
                # * Note only rank the qualified candidates instead of the full list, because of multiple problems (e.g. the same object may get extracted multiple times and we dont want to use set to uniqify them (feel like cheating))
                sim_to_orig[i], outcomes[i] = list(zip(*sorted(zip(sim_to_orig[i], outcomes[i]), key=lambda k: k[0], reverse=True)))
            # Only keep the top n
            outcomes[i] = outcomes[i][0:count]
    else:
        # Filter output based on similarity threshold
        for i in range(len(inputs)):
            debug_message += f"DEBUG: pre-filtered: {outcomes[i]}\n"
            outcomes[i] = [outcomes[i][j] for j in range(len(outcomes[i])) if sim_to_orig[i][j] >= int(vsa.dim * SIM_DETECT_THRESHOLD)]
            # Since we know there'll be no overlapped objects in this workload, we can filter out duplicates
            # Duplicates can appear if one of the objects is more similar to the input than the others (due to biased noise of NN)
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

    if PROFILING:
        prof = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=0, warmup=1, active=PROFILING_SIZE, repeat=1, skip_first=1),
                    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler'),
                    # record_shapes=True,
                    # with_stack=True,
                    # profile_memory=True
                    )
        prof.start()

    rn = Resonator(vsa, mode=VSA_MODE, type=RESONATOR_TYPE, activation=ACTIVATION, act_val=ACT_VALUE, iterations=NUM_ITERATIONS, stoch=STOCHASTICITY, randomness=RANDOMNESS, early_converge=EARLY_CONVERGE, seed=SEED, device=device)

    init_estimates = rn.get_init_estimates().unsqueeze(0).repeat(TEST_BATCH_SIZE,1,1)

    incorrect_count = [0] * MAX_NUM_OBJECTS
    unconverged = [[0,0] for _ in range(MAX_NUM_OBJECTS)]    # [correct, incorrect]
    total_iters = [0] * MAX_NUM_OBJECTS
    max_sim = [[0,0] for _ in range(MAX_NUM_OBJECTS)] # [correct, incorrect]
    min_sim = [[1,1] for _ in range(MAX_NUM_OBJECTS)]
    avg_sim = [[0,0] for _ in range(MAX_NUM_OBJECTS)]
    n = 0


    for images, labels, targets, _ in tqdm(test_dl, desc="Test", leave=True if VERBOSE >= 1 else False):
        if PROFILING:
            prof.step()   # all code between prof.start() and prof.step() are skipped

        images = get_transform()(images.to(device))

        with torch.profiler.record_function("inference"):
            infer_result = model(images)
        
        if QUANTIZE_MODEL:
            # infer_result = (infer_result/ 128.0 * MAX_NUM_OBJECTS).round().type(torch.int8)
            pass
        else:
            # round() will round numbers near 0 to 0, which is not ideal when there's one object, since the vector should be bipolar (either 1 or -1)
            # But 0 is legitimate when there are even number of multiple objects.
            infer_result = infer_result.round().type(vsa.dtype)
        if PROFILING:
            continue

        # Factorization
        # TODO count currently assumes all tests in the batch have the same number of objects. Need to enhance
        outcomes, convergences, iters, counts, debug_message = factorization_algo1(vsa, rn, infer_result, init_estimates, count=len(labels[0]) if COUNT_KNOWN else None)


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

            # Infer result similarty
            infer_sim = get_cos_similarity(infer_result[i], targets[i]).item()

            def collect_sim(val, num_objs, correct):
                idx = 0 if correct else 1
                if val > max_sim[num_objs-1][idx]:
                    max_sim[num_objs-1][idx] = val
                if val < min_sim[num_objs-1][idx]:
                    min_sim[num_objs-1][idx] = val
                avg_sim[num_objs-1][idx] += val

            if incorrect:
                unconverged[len(label)-1][1] += convergence
                incorrect_count[len(label)-1] += 1
                collect_sim(infer_sim, len(label), correct=False)
                if (VERBOSE >= 1):
                    print(Fore.BLUE + f"Test {n} Failed" + Fore.RESET)
                    print("Inference result similarity = {:.3f}".format(infer_sim))
                    print(f"Unconverged: {convergence}")
                    print(f"Iterations: {iter}")
                    print(message[:-1])
                    print(debug_message[:-1])
                    print("Outcome: {}".format(outcome))
            else:
                unconverged[len(label)-1][0] += convergence
                collect_sim(infer_sim, len(label), correct=True)
                if (VERBOSE >= 2):
                    print(Fore.BLUE + f"Test {n} Passed" + Fore.RESET)
                    print("Inference result similarity = {:.3f}".format(infer_sim))
                    print(f"Unconverged: {convergence}")
                    print(f"Iterations: {iter}")
                    print(message[:-1])
                    print(debug_message[:-1])
            n += 1

    if PROFILING:
        prof.stop()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=40))
        func = ["inference", "unbind", "similarity", "activation", "weighted_bundle"]
        for avg in prof.key_averages():
            for f in func:
                if f in avg.key:
                    print(f, avg)
        exit

    for i in range(MAX_NUM_OBJECTS):
        avg_sim[i][0] = "N/A" if incorrect_count[i] == NUM_TEST_SAMPLES_PER_OBJ else round(avg_sim[i][0] / (NUM_TEST_SAMPLES_PER_OBJ - incorrect_count[i]), 3)
        avg_sim[i][1] = "N/A" if incorrect_count[i] == 0 else round(avg_sim[i][1] / incorrect_count[i], 3)
        max_sim[i][0] = round(max_sim[i][0], 3)
        max_sim[i][1] = round(max_sim[i][1], 3)
        min_sim[i][0] = round(min_sim[i][0], 3)
        min_sim[i][1] = round(min_sim[i][1], 3)

    return incorrect_count, unconverged, total_iters, (avg_sim, max_sim, min_sim)

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
        infer_result = infer_result.round().type(vsa.dtype)
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
                    # TODO better to call factorization function with known count, which will call RN multiple times if the extracted object has low similarity
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
                outcome, convergence, iters, count, debug_message = factorization_algo1(vsa, rn, infer_result[0], init_estimates, codebooks=codebooks, known_factor=query)

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
    parser = argparse.ArgumentParser(description="Multi-Concept MNIST")
    parser.add_argument("action", type=str, help="train, test, datagen, eval, reason")
    parser.add_argument("checkpoint", type=str, help="model checkpoint", nargs='?', default=None)
    parser.add_argument("--codebooks", type=str, help="codebook file path", default=None)
    parser.add_argument("--device", type=str, help="device", default=None)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device is not None:
        device = args.device

    print(f"Workload Config: algorithm = {ALGO}, vsa mode = {VSA_MODE}, dim = {DIM}, num pos x = {NUM_POS_X}, num pos y = {NUM_POS_Y}, num color = {NUM_COLOR}, num digits = 10, max num objects = {MAX_NUM_OBJECTS}")

    vsa = get_vsa(test_dir, VSA_MODE, ALGO, args.codebooks, DIM, MAX_NUM_OBJECTS, NUM_COLOR, NUM_POS_X, NUM_POS_Y, FOLD_DIM, EHD_BITS, SIM_BITS, SEED, device)
    model = MultiConceptNonDecomposed(dim=DIM, device=device)

    if args.action == "train":
        print(f"Training on {device}: samples = {NUM_TRAIN_SAMPLES}, epochs = {TRAIN_EPOCH}, batch size = {TRAIN_BATCH_SIZE}")

        if args.checkpoint is not None:
            if os.path.exists(args.checkpoint):
                checkpoint = torch.load(args.checkpoint, map_location=device)
                model.load_state_dict(checkpoint)
                print(f"On top of checkpoint {args.checkpoint}")
            else:
                print("Invalid model checkpoint path.")
                exit(1)

        loss_fn = torch.nn.MSELoss() if ALGO == "algo1" else torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        cur_time_pst = datetime.now().astimezone(timezone('US/Pacific')).strftime("%m-%d-%H-%M")
        train_dl = get_train_data(test_dir, vsa, NUM_TRAIN_SAMPLES, MAX_NUM_OBJECTS, SINGLE_COUNT, TRAIN_BATCH_SIZE, NUM_POS_X, NUM_POS_Y, NUM_COLOR)
        final_loss = train(train_dl, model, loss_fn, optimizer, num_epoch=TRAIN_EPOCH, cur_time=cur_time_pst, device=device)
        model_weight_loc = os.path.join(test_dir, f"model_weights_{MAX_NUM_OBJECTS}objs{'_single_count' if SINGLE_COUNT else ''}_{TRAIN_BATCH_SIZE}batch_{NUM_TRAIN_SAMPLES}samples_{TRAIN_EPOCH}epoch_{final_loss}loss_{cur_time_pst}.pt")
        torch.save(model.state_dict(), model_weight_loc)
        print(f"Model weights saved to {model_weight_loc}")

    # Test mode
    elif args.action == "test":
        print(f"Running test on {device}, batch size = {TEST_BATCH_SIZE}")

        if args.checkpoint and os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint)
        else:
            print("Please provide a valid model checkpoint path.")
            exit(1)

        model.eval()

        print(Fore.CYAN + f"""
Resonator setup:  max_trials = {MAX_TRIALS}, energy_thresh = {ENERGY_THRESHOLD}, similarity_explain_thresh = {SIM_EXPLAIN_THRESHOLD}, \
similarity_detect_thresh = {SIM_DETECT_THRESHOLD}, fold_dim = {FOLD_DIM}, ehd_reg_bits = {EHD_BITS}, sim_reg_bits = {SIM_BITS}, 
resonator = {RESONATOR_TYPE}, iterations = {NUM_ITERATIONS}, stochasticity = {STOCHASTICITY}, randomness = {RANDOMNESS}, \
activation = {ACTIVATION}, act_val = {ACT_VALUE}, early_converge_thresh = {EARLY_CONVERGE}, count_known = {COUNT_KNOWN}
""" + Fore.RESET)

        test_dl = get_test_data(test_dir, vsa, False, NUM_TEST_SAMPLES, MAX_NUM_OBJECTS, SINGLE_COUNT, TEST_BATCH_SIZE, NUM_POS_X, NUM_POS_Y, NUM_COLOR)

        if QUANTIZE_MODEL:
            quan_dl = get_test_data(test_dir, vsa, True, NUM_TEST_SAMPLES if NUM_TEST_SAMPLES < 300 else 300, MAX_NUM_OBJECTS, SINGLE_COUNT, TEST_BATCH_SIZE, NUM_POS_X, NUM_POS_Y, NUM_COLOR)
            quantize_model(model, quan_dl)

        if ALGO == "algo1":
            incorrect_count, unconverged, total_iters, (avg_sim, max_sim, min_sim) = test_algo1(vsa, model, test_dl, device)
        elif ALGO == "algo2":
            incorrect_count, unconverged, total_iters = test_algo2(vsa, model, test_dl, device)

        if SINGLE_COUNT:
            print(f"{MAX_NUM_OBJECTS} objects: Accuracy = {NUM_TEST_SAMPLES - incorrect_count[MAX_NUM_OBJECTS-1]}/{NUM_TEST_SAMPLES}   Unconverged = {unconverged[MAX_NUM_OBJECTS-1]}    Average iterations: {total_iters[MAX_NUM_OBJECTS-1] / (NUM_TEST_SAMPLES)}")
            print(f"{MAX_NUM_OBJECTS} objects: Average similarity = {avg_sim[MAX_NUM_OBJECTS-1]}   Max similarity = {max_sim[MAX_NUM_OBJECTS-1]}    Min similarity = {min_sim[MAX_NUM_OBJECTS-1]}")
        else:
            for i in range(MAX_NUM_OBJECTS):
                print(f"{i+1} objects: Accuracy = {NUM_TEST_SAMPLES//MAX_NUM_OBJECTS - incorrect_count[i]}/{NUM_TEST_SAMPLES//MAX_NUM_OBJECTS}   Unconverged = {unconverged[i]}    Average iterations: {total_iters[i] / (NUM_TEST_SAMPLES//MAX_NUM_OBJECTS)}")
            for i in range(MAX_NUM_OBJECTS):
                print(f"{i+1} objects: Average similarity = {avg_sim[i]}   Max similarity = {max_sim[i]}    Min similarity = {min_sim[i]}")
            
 

    elif args.action == "eval":
        assert TEST_BATCH_SIZE == 1, "Evaluation mode only supports batch size = 1 so far"

        if args.checkpoint and os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
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
                infer_result = infer_result.round().type(vsa.dtype)

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
 
    elif args.action == "reason":
        assert TEST_BATCH_SIZE == 1, "Reason mode only supports batch size = 1 so far"

        if args.checkpoint and os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
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
    elif args.action == "datagen":
        MultiConceptMNIST(test_dir, vsa, train=True, num_samples=NUM_TRAIN_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, single_count=SINGLE_COUNT, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR, force_gen=True)
        MultiConceptMNIST(test_dir, vsa, train=False, num_samples=NUM_TEST_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, single_count=SINGLE_COUNT, num_pos_x=NUM_POS_X, num_pos_y=NUM_POS_Y, num_colors=NUM_COLOR, force_gen=True)
