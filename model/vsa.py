from typing import Literal
from vsa import VSA
import os.path

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