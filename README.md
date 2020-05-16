# MobileV2 Pruning
This repository aim to try out different pruning-approaches on lightweight Backbones.

## Usage
1. Training
    ```
   python main.py --arch MobileNetV2 (for l1norm pruner )
   python main.py --sr --arch MobileNetV2 (for slimming pruner) 
   python main.py --arch USMobileNetV2 (for Autoslim pruner )
    ```
2. Pruning (prune+finetune)
    ```
   python prune.py --arch MobileNetV2 --pruner l1normpruner --pruneratio 0.6
   python prune.py --arch MobileNetV2 --pruner SlimmingPruner --sr --pruneratio 0.6
   python prune.py --arch USMobileNetV2 --pruner AutoSlimPruner
    ```
## Results on Cifar10
|  BackBone| Pruner | Prune Ratio| Original/Pruned/Finetuned Accuracy | FLOPs(M)| Params(M)|
| :---: | :------: |:------: |  :--------------------------: | :-----------------: |:-------------------: |
|MobileV2| L1-Norm|0.6 | 0.937/0.100/0.844|313.5->225.5|2.24->1.15|
|MobileV2| Slimming|0.6 | 0.922/0.485/0.915|313.5->214.5|2.24->0.98|
|MobileV2| AutoSlim|<200 flops| 0.920/0.561/0.916|313.5->199.67|2.24->0.81|
|VGG| Slimming| Optimal Thres|0.926/0.183/0.920 | 399.3->192.67|20.03->1.49|

NOTE:   
1. args for VGG: --arch VGG --s 0.001 --sr --lr 0.02 --epochs 100
## TODO
### Pruning Methodsd
- [x] [l1-norm pruner](https://arxiv.org/abs/1608.08710)
- [x] [Slimming pruner](https://arxiv.org/abs/1708.06519)
- [x] [AutoSlim](https://arxiv.org/abs/1903.11728)
- [ ] ThiNet
- [ ] Soft filter pruning  
**....**
### Backbones
- [x] MobileV2
- [ ] ShuffleNet  
**....**

## Reference
[rethinking-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning) 

[Pruned-MobileNet_v2](https://github.com/eezywu/Pruned-MobileNet_v2) 
