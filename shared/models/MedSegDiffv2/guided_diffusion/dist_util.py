"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8
SETUP_RETRY_COUNT = 3


def setup_dist(args):
    """
    Setup a distributed process group.
    """

    print("Distributed training is disabled.")

    # if dist.is_initialized():
    #     return

    # # Set CUDA devices if multi-GPU is not specified
    # if not args.multi_gpu:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_dev

    # # Force the use of "gloo" backend, as "nccl" and libuv are not supported in this environment
    # backend = "gloo"
    # os.environ["MASTER_ADDR"] = "127.0.0.1"
    # os.environ["RANK"] = os.getenv("RANK", "0")
    # os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", "1")
    # os.environ["MASTER_PORT"] = str(_find_free_port())
    
    # # Initialize the process group
    # dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    mpigetrank = 0  # Set to 0 since MPI is not used here
    if mpigetrank == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    
    return th.load(io.BytesIO(data), weights_only=True, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    """
    Find an available port for distributed communication.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = s.getsockname()[1]
    s.close()
    return port