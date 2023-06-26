from subprocess import Popen
import argparse
import time
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--clients', type=int)
parser.add_argument('--gpus', nargs='+')

args = parser.parse_args()

run_id = str(round(time.time() * 1000))
checkpoint_dir = f'./checkpoints/chk_{run_id}'

# prepare checkpoint directory
os.makedirs(checkpoint_dir, exist_ok=True)
shutil.copyfile('./config.py', os.path.join(checkpoint_dir, 'config.py'))

processes = []
for c in range(args.clients):
    os.mkdir(os.path.join(checkpoint_dir, f'client_{c}'))
    gpu_idx = c % len(args.gpus)
    process = Popen(['python', 'client.py', '--gpu', str(args.gpus[gpu_idx]), '--id', str(c), '--checkpoint-dir', os.path.join(checkpoint_dir, f'client_{c}')])
    processes.append(process)

for p in processes:
    p.wait()