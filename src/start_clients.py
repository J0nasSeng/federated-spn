from subprocess import Popen
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--clients', type=int)
parser.add_argument('--gpus', nargs='+')

args = parser.parse_args()

processes = []
for c in range(args.clients):
    gpu_idx = c % len(args.gpus)
    process = Popen(['python', 'client.py', '--gpu', str(args.gpus[gpu_idx]), '--id', str(c)])
    processes.append(process)

for p in processes:
    p.wait()