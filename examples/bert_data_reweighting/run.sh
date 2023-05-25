# Baseline
python main.py --batch_size 16 --baseline --precision fp16 --seed 0
python main.py --batch_size 16 --baseline --precision fp16 --seed 1
python main.py --batch_size 16 --baseline --precision fp16 --seed 2
# Meta-Weight-Net (single GPU fp 16)
python main.py --batch_size 16 --precision fp16 --seed 0
python main.py --batch_size 16 --precision fp16 --seed 1
python main.py --batch_size 16 --precision fp16 --seed 2

# Meta-Weight-Net (Multi GPU + ZeRO)
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --batch_size 8 --precision fp16 --strategy zero --seed 0
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --batch_size 8 --precision fp16 --strategy zero --seed 1
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --batch_size 8 --precision fp16 --strategy zero --seed 2

