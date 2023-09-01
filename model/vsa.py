# %%
from torchhd.types import VSAOptions
import os.path
import torchhd as hd
import torch
from torchvision.datasets import utils
from typing import List, Tuple
import random

# %%
class VSA:
    # codebooks for each factor
    codebooks: List[hd.VSATensor] or hd.VSATensor

    def __init__(
            self,
            root: str,
            dim: int,
            model:VSAOptions,
            num_factors: int,
            num_codevectors: int or Tuple[int], # number of vectors per factor, or tuple of number of codevectors for each factor
            seed: None or int = None,  # random seed
            device = "cpu"
        ):

        self.root = root
        self.model = model
        self.device = device
        # # MAP default is float, we want to use int
        # if model == 'MAP':
        #     self.dtype = torch.int8
        # else:
        self.dtype = None
        self.dim = dim
        self.num_factors = num_factors
        self.num_codevectors = num_codevectors

        if self._check_exists("codebooks.pt"):
            self.codebooks = torch.load(os.path.join(self.root, "codebooks.pt"))
        else:
            self.codebooks = self.gen_codebooks(seed)


    def gen_codebooks(self, seed) -> List[hd.VSATensor] or hd.VSATensor:
        if seed is not None:
            torch.manual_seed(seed)
        l = []
        # All factors have the same number of vectors
        if (type(self.num_codevectors) == int):
            for i in range(self.num_factors):
                l.append(hd.random(self.num_codevectors, self.dim, vsa=self.model, dtype=self.dtype, device=self.device))
            l = torch.stack(l).to(self.device)
        # Every factor has a different number of vectors
        else:
            for i in range(self.num_factors):
                l.append(hd.random(self.num_codevectors[i], self.dim, vsa=self.model, dtype=self.dtype, device=self.device))
            
        os.makedirs(self.root, exist_ok=True)
        torch.save(l, os.path.join(self.root, f"codebooks.pt"))

        return l
    
    def sample(self, num_samples, num_vectors_supoerposed = 1, noise=0.0):
        '''
        Sample `num_samples` random vectors from the dictionary, or multiple vectors superposed
        '''
        labels = [None] * num_samples
        vectors = hd.empty(num_samples, self.dim, vsa=self.model, dtype=self.dtype, device=self.device)
        for i in range(num_samples):
            labels[i]= [tuple([random.randint(0, len(self.codebooks[i])-1) for i in range(self.num_factors)]) for j in range(num_vectors_supoerposed)]
            vectors[i] = self.apply_noise(self.__getitem__(labels[i]), noise)
        return labels, vectors

    def apply_noise(self, vector, noise):
        # orig = vector.clone()
        indices = [random.random() < noise for i in range(self.dim)]
        vector[indices] = self.flip(vector[indices])
        
        # print("Verify noise:" + str(hd.dot_similarity(orig, vector)))
        return vector.to(self.device)
    
    def flip(self, vector):
        if (self.model == 'MAP'):
            return -vector
        elif (self.model == "BSC"):
            return 1 - vector

    def cleanup(self, inputs):
        '''
        input: `(b, f, d)` :tensor. b is batch size, f is number of factors, d is dimension
        Return: List[Tuple(int)] of length b
        '''
        if type(self.codebooks) == list:
            winners = torch.empty((inputs.size(0), self.num_factors), dtype=torch.int8, device=self.device)
            for i in range(self.num_factors):
                winners[:,i] = torch.argmax(torch.abs(self.similarity(inputs[:,i], self.codebooks[i])), -1)
            return [tuple(winners[i].tolist()) for i in range(winners.size(0))]
        else:
            winners = torch.argmax(torch.abs(self.similarity(inputs.unsqueeze(-2), self.codebooks).squeeze(-2)), -1)
            return [tuple(winners[i].tolist()) for i in range(winners.size(0))]
      
    def similarity(self, input: hd.VSATensor, others: hd.VSATensor) -> hd.VSATensor:
        """Inner product with other hypervectors.
        Shapes:
            - input: :math:`(*, d)`
            - others: :math:`(n, d)` or :math:`(d)`
            - output: :math:`(*, n)` or :math:`(*)`, depends on shape of others
        """
        if isinstance(input, hd.MAPTensor):
            return input.dot_similarity(others)
        elif isinstance(input, hd.BSCTensor):
            return hd.hamming_similarity(input, others)


    def ensure_vsa_tensor(self, data):
        return hd.ensure_vsa_tensor(data, vsa=self.model, dtype=self.dtype, device=self.device)

    def get_vector(self, key:tuple):
        '''
        `key` is a tuple of indices of each factor
        Instead of pre-generate the dictionary, we combine factors to get the vector on the fly
        This saves meomry, and also the dictionary lookup is only used during sampling and comparison
        '''
        assert(len(key) == self.num_factors)
        factors = [self.codebooks[i][key[i]] for i in range(self.num_factors)]
        return hd.multibind(torch.stack(factors)).to(self.device)

    def __getitem__(self, key: list):
        '''
        `key` is a list of tuples in [(f0, f1, f2, ...), ...] format.
        fx is the index of the factor in the codebook, which is also its label.
        '''
        if (len(key) == 1):
            return self.get_vector(key[0])
        else:
            # TODO to be tested
            return hd.multiset(torch.stack([self.get_vector(key[i]) for i in range(len(key))]))
    
 
    def _check_exists(self, file) -> bool:
        return utils.check_integrity(os.path.join(self.root, file))
# %%

class MultiConceptMNISTVSA(VSA):

    def __init__(
            self,
            root: str,
            dim: int = 2048,
            model:VSAOptions = 'MAP',
            max_num_objects = 3,
            num_pos_x = 3,
            num_pos_y = 3,
            num_colors = 7,
            seed: None or int = None,  # random seed
            device = "cpu"):

        super().__init__(root, dim, model, num_factors = 4, num_codevectors = (num_pos_x, num_pos_y, num_colors, 10), seed = seed, device = device)

        # Default is float, we want to use int
        if model == 'MAP':
            self.dtype = torch.int8
        else:
            self.dype = None

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