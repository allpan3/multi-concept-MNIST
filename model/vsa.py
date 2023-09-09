from typing import Literal
from vsa import VSA
import itertools
import torch

class MultiConceptMNISTVSA1(VSA):

    def __init__(
            self,
            root: str,
            model: Literal['SOFTWARE', 'HARDWARE'] = 'SOFTWARE',
            dim: int = 2048,
            max_num_objects = 3,
            num_pos_x = 3,
            num_pos_y = 3,
            num_colors = 7,
            seed: None or int = None,  # random seed
            device = "cpu"):

        super().__init__(root, model, dim, num_factors = 4, num_codevectors = (num_pos_x, num_pos_y, num_colors, 10), seed = seed, device = device)

        self.num_pos_x = num_pos_x
        self.num_pos_y = num_pos_y
        self.num_colors = num_colors


    def lookup(self, label: list):
        '''
        `label` is a list of dict in [{'pos_x': int, 'pos_y': int, 'color': int, 'digit': int}, ...] format
        '''
        key = []
        for i in range(len(label)):
            key.append((label[i]["pos_x"], label[i]["pos_y"], label[i]["color"], label[i]["digit"]))
        return self.__getitem__(key)


class MultiConceptMNISTVSA2(VSA):

    def __init__(
        self,
        root: str,
        model: Literal['SOFTWARE', 'HARDWARE'] = 'SOFTWARE',
        dim: int = 2048,
        max_num_objects = 3,
        num_pos_x = 3,
        num_pos_y = 3,
        num_colors = 7,
        seed: None or int = None,  # random seed
        device = "cpu"):

        # Assign an ID to each object in the scene
        self.num_id = max_num_objects
        self.num_pos_x = num_pos_x
        self.num_pos_y = num_pos_y
        self.num_colors = num_colors

        super().__init__(root, model, dim, num_factors = 5, num_codevectors = (num_pos_x, num_pos_y, num_colors, 10, self.num_id), seed = seed, device = device)

        self.id_codebook = self.codebooks[-1]
        self.x_codebook = self.codebooks[0]
        self.y_codebook = self.codebooks[1]

    def lookup(self, label: list, bundled: bool = True):
        '''
        `label` is a list of dict in [{'pos_x': int, 'pos_y': int, 'color': int, 'digit': int}, ...] format
        We reorder the label list based on the (x, y) locations of the objects in the scene, then bind the corresponding
        ID to the compositional vector
        '''
        key = []
        for i in range(len(label)):
            # Label without ID
            key.append((label[i]["pos_x"], label[i]["pos_y"], label[i]["color"], label[i]["digit"]))
            
        # Get all objects (excluding ID)
        objects = [self.__getitem__(key[j]) for j in range(len(key))]

        # Construct priority list for sorting based on (x, y) locations
        rule = [x for x in itertools.product(range(len(self.x_codebook)), range(len(self.y_codebook)))]
        # Reorder the positions of the objects in each label in the ascending order of (x, y), the first two elements in the label
        _, objects = list(zip(*sorted(zip(key, objects), key=lambda k: rule.index(k[0][0:2]))))
        # Remember the original indice of the codebooks for reordering later
        indices = sorted(range(len(key)), key=lambda k: rule.index(key[k][0:2]))
        # Bind the vector with ID determined by the position in the list
        objects = [self.bind(objects[j], self.id_codebook[j]) for j in range(len(objects))]
        # Return to the original order (for similarity check)
        objects = [objects[i] for i in indices]
        if bundled:
            objects = self.multiset(torch.stack(objects))

        return objects



