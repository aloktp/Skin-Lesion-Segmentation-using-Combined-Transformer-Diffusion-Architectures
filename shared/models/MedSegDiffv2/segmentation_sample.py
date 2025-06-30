import os
import sys
import json
import random
import itertools
import threading
import time
from datetime import timedelta
from collections import OrderedDict
import argparse
import numpy as np
import torch as th
import torchvision.utils as vutils
from torchvision import transforms
from guided_diffusion import dist_util
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import logging

# Set up the logger
log_dir = os.getenv("APP_LOG_PATH", "/shared/logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

stop_spinner = False

def spinner():
    global stop_spinner
    for symbol in itertools.cycle(['⠋', '⠙', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇']):
        if stop_spinner:
            break
        sys.stdout.write(f'\r{symbol} Processing... ')
        sys.stdout.flush()
        time.sleep(0.1)

seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img
    
def main():
    global stop_spinner

    args = create_argparser().parse_args()

    # Set device to CPU if CUDA is unavailable
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dist_util.setup_dist(args)
    logger.info("Distributed training is disabled.")

    # Log the output directory for debugging
    os.makedirs(args.out_dir, exist_ok=True)
    logger.info(f"Output directory: {args.out_dir}")

    spinner_thread = threading.Thread(target=spinner, daemon=True)
    spinner_thread.start()
    
    logger.info("Setting up data transformations...")
    tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
    transform_test = transforms.Compose(tran_list)

    logger.info("Loading dataset...")
    ds = ISICDataset(args, args.data_dir, transform_test)
    args.in_ch = 4
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)
    
    logger.info("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )



    logger.info("Loading model checkpoint...")
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")

    # Handle 'module.' prefix in checkpoint keys
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "module." in k:
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v  # Keep the key as is

    model.load_state_dict(new_state_dict)
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.info("Starting sampling process...")
    num_samples = len(data)
    logger.info(f"Number of samples to process: {num_samples}")

    total_time = 0
    for i, (b, m, path) in enumerate(data, start=1):
        loop_start_time = time.time()
        time.sleep(0.05)

        # Move tensors to the selected device
        b = b.to(device)
        c = th.randn_like(b[:, :1, ...], device=device)
        img = th.cat((b, c), dim=1)

        slice_ID = path[0].split("_")[-1].split('.')[0]
        logger.info(f"Processing sample {i}/{num_samples}, ID: {slice_ID}")
        # start_event = th.cuda.Event(enable_timing=True)
        # end_event = th.cuda.Event(enable_timing=True)

        enslist = []

        for j in range(args.num_ensemble):

            model_kwargs = {}
            # Use timing based on available device
            if th.cuda.is_available():
                start_event = th.cuda.Event(enable_timing=True)
                end_event = th.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                img,
                step=args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            if th.cuda.is_available():
                end_event.record()
                th.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)
            else:
                elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

            logger.info(f"Sample {j + 1} time: {elapsed_time:.2f} ms")  

            co = th.tensor(cal_out, device=device)
            if args.version == 'new':
                enslist.append(sample[:, -1, :, :])
            else:
                enslist.append(co)
                
            if args.debug:
                if args.data_name == 'ISIC':
                    o = th.tensor(org)[:,:-1,:,:]
                    c = th.tensor(cal).repeat(1, 3, 1, 1)

                    s = sample[:,-1,:,:]
                    b,h,w = s.size()
                    ss = s.clone()
                    ss = ss.view(s.size(0), -1)
                    ss -= ss.min(1, keepdim=True)[0]
                    ss /= ss.max(1, keepdim=True)[0]
                    ss = ss.view(b, h, w)
                    ss = ss.unsqueeze(1).repeat(1, 3, 1, 1)

                    tup = (ss,o,c)
                compose = th.cat(tup,0)
                vutils.save_image(compose, fp = os.path.join(args.out_dir, str(slice_ID)+'_output'+str(i)+".jpg"), nrow = 1, padding = 10)

        ensres = staple(th.stack(enslist, dim=0)).squeeze(0)
        output_path = os.path.join(args.out_dir, f"{slice_ID}_output_ens.jpg")
        vutils.save_image(ensres, fp=output_path, nrow=1, padding=10)
        logger.info(f"Saved ensemble output image: {output_path}")

        loop_time = time.time() - loop_start_time
        total_time += loop_time
        avg_time_per_loop = total_time / i
        remaining_time = avg_time_per_loop * (num_samples - i)
        logger.info(f"Sample {i}/{num_samples} completed. Remaining time: {timedelta(seconds=int(remaining_time))}")

    stop_spinner = True
    spinner_thread.join()
    sys.stdout.write('\r✔ Process Completed!\n')
    sys.stdout.flush()


def create_argparser():
    defaults = dict(
        data_name = 'ISIC',
        clip_denoised=True,
        data_dir='',
        out_dir='', 
        model_path='', 
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        num_ensemble=5,      #number of samples in the ensemble
        gpu_dev = "0",
        multi_gpu = None, #"0,1,2"
        debug = False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
