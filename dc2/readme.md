```bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 main.py --batch-size 32 --epochs 20
```