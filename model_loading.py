import os

import torch
import torch.distributed as dist


# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#
#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)
#
# def cleanup():
#     dist.destroy_process_group()
#
# setup(0, 8)



path = "/home/szzhao/LT_project/dinov2/output/eval/training_56249/teacher_checkpoint.pth"

model = torch.load(path, map_location='cpu')['teacher']

print(model.keys())