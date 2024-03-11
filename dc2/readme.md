```bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 main.py --batch-size 32 --epochs 20
```

```bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train_xlnet.py --batch-size 8 --epochs 30 --lr=0.000002 
```


```bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train_roberta.py --batch-size 32 --epochs 20 --lr=0.000005 
```


```bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 train_roberta.py --batch-size 32 --epochs 20 --lr=0.000005 
```