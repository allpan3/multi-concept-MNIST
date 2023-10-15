"""Add {PROJECT_ROOT}. to PYTHONPATH

Usage:
import this module before import any intra-project modules
e.g 
    import _init_paths
    from datasets import dataset
""" 

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

project_root = osp.abspath(osp.dirname(osp.dirname(__file__)))

# Add project dir to PYTHONPATH
add_path(project_root)