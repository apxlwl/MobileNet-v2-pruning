# MobileV2 Pruning
This repository aim to try out different pruning-approaches on lightweight Backbones.

## Usage
1. Training (about with ReduceLROnPlateau scheduler)
    ```
   python main.py --sr(for slimming) 
    ```
2. Pruning (prune+finetune)
    ```
   python prune.py --pruner l1normpruner --pruneratio 0.2
   python prune.py --pruner SlimmingPruner --sr --pruneratio 0.2
    ```
## Results on Cifar10
|  BackBone| Pruner | Prune Ratio| Original/Pruned/Finetuned Accuracy | FLOPs(M)| Params(M)|
| :---: | :------: |:------: |  :--------------------------: | :-----------------: | :-------------------: |
|MobileV2| L1-Norm|0.2 | 0.937/0.919/0.935|313.5->281.9|2.24->1.87|
|MobileV2| L1-Norm|0.4 | 0.937/0.504/0.918|313.5->250.6|2.24->1.51|
|MobileV2| L1-Norm|0.6 | 0.937/0.100/0.844|313.5->225.5|2.24->1.15|
|MobileV2| Slimming|0.2 | 0.922/0.891/0.920|313.5->277.2|2.24->1.82|
|MobileV2| Slimming|0.4 | 0.922/0.865/0.920|313.5->244.5|2.24->1.41|
|MobileV2| Slimming|0.6 | 0.922/0.686/0.915|313.5->214.5|2.24->0.98|

## TODO
### Pruning Methodsd
- [x] l1-norm pruner
- [x] Slimming pruner
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