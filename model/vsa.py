# %%
import torch
import torchhd
from torchhd.types import VSAOptions
from torchhd import bind, bundle
import itertools

class MultiConceptMNISTVSA:
    # Dictionary of compositional vectors of the objects
    # Only single objects are stored. Multiple objects are composed on the fly due to large memoery requirement
    dict = {}

    def __init__(
            self,
            dim: int = 2048,
            vsa:VSAOptions = 'MAP',
            max_num_objects = 3,
            num_pos_x = 3,
            num_pos_y = 3,
            num_colors = 7
            ):

        # Default is float, we want to use int
        if vsa == 'MAP':
            dtype = torch.int8
        else:
            dype = None

        self.dim = dim
        self.num_pos_x = num_pos_x
        self.num_pos_y = num_pos_y
        self.num_colors = num_colors
        
        self.pos_x = torchhd.random(num_pos_x, dim, vsa=vsa, dtype=dtype)
        self.pos_y = torchhd.random(num_pos_y, dim, vsa=vsa, dtype=dtype)
        self.color = torchhd.random(num_colors, dim, vsa=vsa, dtype=dtype)
        self.digit = torchhd.random(10, dim, vsa=vsa, dtype=dtype)

        self.gen_dict()
    
    def gen_dict(self):
        '''
        Generate dictionary of all possible combinations of a single object
        key is tuple of (pos_x, pos_y, color, digit)
        value is the tensor of the object
        '''
        for x, y, c, d in itertools.product(range(len(self.pos_x)), range(len(self.pos_y)), range(len(self.color)), range(len(self.digit))):
            self.dict[(x, y, c, d)] = bind(bind(self.pos_x[x], self.pos_y[y]), bind(self.color[c], self.digit[d]))

    def __getitem__(self, key: list):
        '''
        `key` is a list of tuples of (pos_x, pos_y, color, digit). The number of tuples represents the number of objects
        '''
        if (len(key) == 1):
            return self.dict[tuple(key[0])]
        else:
            obj = self.dict[tuple(key[0])]
            i = 1
            while i < len(key):
                obj = bundle(obj, self.dict[tuple(key[i])])
                i += 1
            return obj 