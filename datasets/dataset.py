""" Multi-Concept MNIST Dataset
"""

from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets.vision import VisionDataset
import torch
import random
import json
import os
from vsa import VSA
from typing import Callable, Optional
import itertools
from tqdm import tqdm

class MultiConceptMNIST(VisionDataset):

    COLOR_SET = [
        range(0,3),   # white
        range(0,1),   # red
        range(1,2),   # green
        range(2,3),   # blue
        range(0,2),   # yellow
        range(0,3,2), # magenta
        range(1,3)    # cyan
    ]

    def __init__(
        self,
        root: str,
        vsa: VSA = None,
        train: bool = False,      # training set or test set
        num_samples: int = 1000,
        force_gen: bool = False,  # generate dataset even if it exists
        max_num_objects: int = 3,
        single_count: bool = False,   # using only `max_num_objects` for training (instead of a range)
        num_pos_x: int = 3,
        num_pos_y: int = 3,
        num_colors: int = 7,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if (train):
            # Allocate more samples for larger object counts for training
            total = sum([x+1 for x in range(max_num_objects)])
            if not single_count:
                num_samples = [round((x+1)/total * num_samples) for x in range(max_num_objects)]
        else:
            if not single_count:
                # Even sample numbers for testing
                num_samples = [num_samples // max_num_objects] * max_num_objects

        if vsa:
            assert(vsa.num_pos_x == num_pos_x)
            assert(vsa.num_pos_y == num_pos_y)
            assert(vsa.num_colors == num_colors)
            self.vsa = vsa

        self.num_pos_x = num_pos_x
        self.num_pos_y = num_pos_y
        self.num_colors = num_colors
        assert(max_num_objects <= num_pos_x * num_pos_y)
        self.max_num_objects = max_num_objects

        if transform is None:
            self.transform = lambda x: x
 
        self.data = []
        self.labels = []
        self.targets = []
        self.questions = []

        type = "train" if train else "test"

        # Generate dataset if not exists
        if single_count:
            if force_gen or not self._check_exists(type, max_num_objects, num_samples):
                assert vsa is not None, "VSA model must be provided for dataset generation"
                raw_ds = MNIST(root=os.path.join(self.root, "../.."), train=train, download=True)
                self.data, self.labels, self.targets, self.questions = self.dataset_gen(type, raw_ds, max_num_objects, num_samples)
            else:
                print(f"{type} {max_num_objects} obj {num_samples} dataset exists, loading...")
                self.data = self._load_data(os.path.join(self.root, f"{type}-images-{max_num_objects}obj-{num_samples}samples.pt"))
                self.labels = self._load_json(os.path.join(self.root, f"{type}-labels-{max_num_objects}obj-{num_samples}samples.json"))
                self.targets = self._load_data(os.path.join(self.root, f"{type}-targets-{max_num_objects}obj-{num_samples}samples.pt"))
                if not train:
                    self.questions = self._load_json(os.path.join(self.root, f"{type}-questions-{max_num_objects}obj-{num_samples}samples.json"))
        else:
            for i in range(0, self.max_num_objects):
                n = i + 1
                if force_gen or not self._check_exists(type, n, num_samples[i]):
                    assert vsa is not None, "VSA model must be provided for dataset generation"
                    raw_ds = MNIST(root=os.path.join(self.root, "../.."), train=train, download=True)
                    data, labels, targets, questions = self.dataset_gen(type, raw_ds, n, num_samples[i])
                    self.data += data
                    self.labels += labels
                    self.targets += targets
                    self.questions += questions
                else:
                    print(f"{type} {n} obj {num_samples[i]} dataset exists, loading...")
                    self.data += self._load_data(os.path.join(self.root, f"{type}-images-{n}obj-{num_samples[i]}samples.pt"))
                    self.labels += self._load_json(os.path.join(self.root, f"{type}-labels-{n}obj-{num_samples[i]}samples.json"))
                    self.targets += self._load_data(os.path.join(self.root, f"{type}-targets-{n}obj-{num_samples[i]}samples.pt"))
                    if not train:
                        self.questions += self._load_json(os.path.join(self.root, f"{type}-questions-{n}obj-{num_samples[i]}samples.json"))
        

    def _check_exists(self, type: str, num_obj, num_samples) -> bool:
        if type == "train":
            return all(
                os.path.exists(os.path.join(self.root, file))
                for file in [f"{type}-{t}-{num_obj}obj-{num_samples}samples.pt" for t in ["targets", "images"]]
            )
        else:
            return all(
                [os.path.exists(os.path.join(self.root, file))
                 for file in [f"{type}-{t}-{num_obj}obj-{num_samples}samples.pt" for t in ["targets", "images"]]]
                 + [os.path.exists(os.path.join(self.root, file))
                    for file in [f"{type}-{t}-{num_obj}obj-{num_samples}samples.json" for t in ["questions"]]]
            )

    def _load_json(self, path: str):
        with open(path, "r") as path:
            return json.load(path)

    def _load_data(self, path: str):
        return torch.load(path)

    def __getitem__(self, index: int):
        if self.train:
            return self.data[index], self.labels[index], self.targets[index]
        else:
            return self.data[index], self.labels[index], self.targets[index], self.questions[index]

    def __len__(self) -> int:
        return len(self.data)

    def dataset_gen(self, type: str, raw_ds: MNIST, n_obj, num_samples):
        assert(self.num_colors <= len(self.COLOR_SET))

        os.makedirs(self.root, exist_ok=True)

        image_set, label_set = self.image_gen(num_samples, n_obj, raw_ds)
        print("Saving images...", end="", flush=True)
        torch.save(image_set, os.path.join(self.root, f"{type}-images-{n_obj}obj-{num_samples}samples.pt"))
        print("Done. Saving labels...", end="", flush=True)
        with open(os.path.join(self.root, f"{type}-labels-{n_obj}obj-{num_samples}samples.json"), "w") as f:
            json.dump(label_set, f)
        print("Done.")
        target_set = self.target_gen(label_set)
        print("Saving targets...", end="", flush=True)
        torch.save(target_set, os.path.join(self.root, f"{type}-targets-{n_obj}obj-{num_samples}samples.pt"))
        print("Done.")

        question_set = []
        if (type == "test"):
            question_set = self.question_gen(label_set) 
            print("Saving questions...", end="", flush=True)
            with open(os.path.join(self.root, f"{type}-questions-{n_obj}obj-{num_samples}samples.json"), "w") as f:
                json.dump(question_set, f)
            print("Done.")

        return image_set, label_set, target_set, question_set

    def image_gen(self, num_samples, num_objs, raw_ds: MNIST) -> [torch.Tensor, list]:
        image_set = []
        label_set_uni = set() # Check the coverage of generation
        label_set = []

        for i in tqdm(range(num_samples), desc=f"Generating {num_samples} samples of {num_objs}-object images", leave=False):
            # num_pos_x by num_pos_y grid, each grid 28x28 pixels, color (3-dim uint8 tuple)
            # [C, H, W]
            image_tensor = torch.zeros(3, 28*self.num_pos_y, 28*self.num_pos_x, dtype=torch.uint8)
            label = []
            label_uni = set() # Check the coverage of generation

            # one object max per position
            all_pos = [x for x in itertools.product(range(self.num_pos_x), range(self.num_pos_y))]
            for j in range(num_objs):
                pick = random.randint(0,len(all_pos)-1)
                pos_x, pos_y = all_pos.pop(pick)
                color_idx = random.randint(0, self.num_colors-1)
                mnist_idx = random.randint(0, 59999 if raw_ds.train else 9999)
                digit_image = raw_ds.data[mnist_idx, :, :]
                for k in self.COLOR_SET[color_idx]:
                    image_tensor[k, pos_y*28:(pos_y+1)*28, pos_x*28:(pos_x+1)*28] = digit_image

                image_tensor = self.transform(image_tensor)
                digit = raw_ds.targets[mnist_idx].item()
                object = {
                    "pos_x": pos_x, 
                    "pos_y": pos_y,
                    "color": color_idx,
                    "digit": digit
                }
                label.append(object)
                # For coverage check. Since pos is checked to be unique, objects are unique
                label_uni.add((pos_x, pos_y, color_idx, digit))

            # For coverage check, convert to tuple to make it hashable and unordered
            label_set_uni.add(tuple(label_uni)) 
            label_set.append(label)
            image_set.append(image_tensor)

        print(f"Generated {num_samples} samples of {num_objs}-object images. Coverage: {len(label_set_uni)} unique labels.")
        return image_set, label_set  

    def target_gen(self, label_set: list or str) -> list:
        if type(label_set) == str:
            with open(label_set, "r") as f:
                label_set = json.load(f)
        
        target_set = []
        for label in tqdm(label_set, desc=f"Generating targets for {len(label_set[0])}-object images", leave=False):
            target_set.append(self.vsa.lookup(label))
        return target_set

    QUESTION_SET = [
        "object_exists", "object_count", "object_of_same_type_exists"
    ]

    def question_gen(self, label_set: list or str) -> list:
        if type(label_set) == str:
            with open(label_set, "r") as f:
                label_set = json.load(f)

        attr_nums = [self.num_pos_x, self.num_pos_y, self.num_colors, 10]
        question_set = []
        for _label in tqdm(label_set, desc=f"Generating questions for {len(label_set[0])}-object images", leave=False):
            # TODO we may just save label in tuple format since the dict doesn't really provide additional information. Or at least pass labels to target and question gen in tuple format directly
            label = []
            for i in range(len(_label)):
                label.append((_label[i]["pos_x"], _label[i]["pos_y"], _label[i]["color"], _label[i]["digit"]))
            questions = []
            # Generate a question and its answer for each question type
            for q in self.QUESTION_SET:
                # Pick query based on the question type (not all queries are valid for all questions)
                if q == "object_of_same_type_exists":
                    # Randomly pick an object in the label
                    query = list(random.choice(label))
                    # Randomly decide whether to include color, digit, or both
                    pick = random.choice([(True, True), (True, False), (False, True)])
                    for i in [2, 3]:
                        query[i] = query[i] if pick[i-2] else None
                    query = tuple(query)
                    prompt_ = "color and digit" if pick[0] and pick[1] else "color" if pick[0] else "digit" if pick[1] else ""
                    prompt = f"Is there any object with the same {prompt_} as the object at position {query[0], query[1]} in the scene?"
                elif q == "object_count":
                    # Count the number of objects matching the color and/or digit
                    pick = random.choice([(True, True), (True, False), (False, True)])
                    query = [None] * 4
                    for i in [2, 3]:
                        if pick[i-2]:
                            query[i] = random.randint(0, attr_nums[i]-1)
                    query = tuple(query)
                    prompt = f"How many objects of" + \
                             ((f" color = {query[2]}," if query[2] != None else "") + \
                             (f" digit = {query[3]}," if query[3] != None else ""))[:-1] + \
                             " are there in the scene?"
                elif q == "object_exists":
                    # Any query is fine. Randomlly generate one
                    query = [None] * 4
                    # Make sure at least one attribute is not None
                    while (all([x == None for x in query])):
                        for i in range(4):
                            query[i] = random.choice([True, None])
                            if query[i]:
                                query[i] = random.randint(0, attr_nums[i]-1)
                    query = tuple(query)
                    prompt = f"Does an object with" + \
                             ((f" x = {query[0]}," if query[0] != None else "") + \
                             (f" y = {query[1]}," if query[1] != None else "") + \
                             (f" color = {query[2]}," if query[2] != None else "") + \
                             (f" digit = {query[3]}," if query[3] != None else ""))[:-1] + \
                             " exist?"

                answer = getattr(self, q)(label, query)
                questions.append({
                    "question": q,
                    "prompt": prompt,
                    "query": query,
                    "answer": answer
                })
            
            question_set.append(questions)

        return question_set

    def _match(self, object: tuple, query: tuple) -> bool:
        for i in range(len(object)):
            if query[i] != None and object[i] != query[i]:
                return False
        return True

    def object_exists(self, label, query: tuple) -> bool:
        for object in label:
            if self._match(object, query):
                return True
        
        return False
    
    def object_count(self, label, query: tuple) -> int:
        count = 0
        for object in label:
            if self._match(object, query):
                count += 1
        
        return count

    def object_of_same_type_exists(self, label, query: tuple) -> bool:
        """
        Find if an object with the same color and/or digit as the query exists
        """
        # Get rid of positions (as they are not part of the query)
        _query = (None, None, *query[2:])
        for object in label:
                # Make sure it's not the query itself
            if query[:2] != object[:2] and self._match(object, _query):
                return True
        return False