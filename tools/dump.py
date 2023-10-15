#%%
import _init_paths   # pylint: disable=unused-import
from datasets import get_test_data
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from contextlib import redirect_stdout
import torchvision.transforms as transforms
from model_quantization import quantize_model
from models.nn_non_decomposed import MultiConceptNonDecomposed
import sys
import os.path

# def imshow(img):
#     plt.imshow(img.permute(1,2,0))
#     plt.show()

def imshow(img):
    # img = img / 2 + 0.5     # unnormalize (from [-1,1] to [0,1])
    plt.imshow(img.permute(1,2,0))
    plt.show()

def print_tensor(t: torch.Tensor):
    if len(t.shape) == 0:
        print(t.item(), end="")
        return
    
    print("{", end="")
    
    for i,m in enumerate(t):
        print_tensor(m)

        if i < len(t) - 1:
            print(",", end="")

    print("}", end="")

def generate_image_header(images: Tensor, scaling_factor: int = 1):
    """
    images: Tensor of shape (N, C, H, W)
    """
    
    print(r'''#ifndef MC_MNIST_IMAGES_H
#define MC_MNIST_IMAGES_H
''')

    shape = images.shape
    print(f"static const elem_t images[{shape[0]}][{shape[1]}][{shape[2]}][{shape[3]}] row_align(1) = ", end="")

    scaled_images = torch.clamp(images * scaling_factor, min=-128, max=127).round().int()
    print_tensor(scaled_images)

    print(";\n\n#endif\n")


if __name__ == "__main__":
    NUM_SAMPLES = 100
    MAX_NUM_OBJECTS = 3
    SINGLE_COUNT = True
    DIM = 1024

    test_dir = "./tests/HARDWARE-1024dim-256fd-3x-3y-7color/algo1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiConceptNonDecomposed(dim=DIM, device=device)

    if sys.argv[-1].endswith(".pt"):
        if os.path.exists(sys.argv[-1]):
            checkpoint = torch.load(sys.argv[-1], map_location=device)
            model.load_state_dict(checkpoint)
            print(f"On top of checkpoint {sys.argv[-1]}")
        else:
            print("Invalid model checkpoint path.")
            exit(1)
    transform = transforms.Compose([
            # transforms.Resize(224, antialiased = True),
            transforms.ConvertImageDtype(torch.float32)  # Converts to [0, 1]
        ])

    dl = get_test_data(root=test_dir, num_samples=NUM_SAMPLES, max_num_objects=MAX_NUM_OBJECTS, single_count=SINGLE_COUNT, batch_size=1)

    images = transform(dl.dataset.data)

    quantize_model(model, dl)
    # with open('images.h', 'w') as f:
    #     with redirect_stdout(f):
    #         generate_image_header(ds.data, quantize=False)

# %%
