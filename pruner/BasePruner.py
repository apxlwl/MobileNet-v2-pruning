import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim


class BasePruner:
    def __init__(self, model, newmodel, testset, trainset, optimizer,args):
        self.model = model
        self.newmodel = newmodel
        self.testset = testset
        self.trainset = trainset
        self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3,threshold=1e-2)
        self.args=args
    def prune(self):
        raise NotImplementedError

    def test(self, newmodel=True,ckpt=None):
        if newmodel:
            model = self.newmodel
        else:
            model = self.model
        if ckpt:
            model.load_state_dict(ckpt)
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in tqdm(self.testset, total=len(self.testset)):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.testset.dataset)
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        #     test_loss, correct, len(self.testset.dataset),
        #     100. * correct / len(self.testset.dataset)))
        return correct.item() / float(len(self.testset.dataset))

    def train(self):
        self.newmodel.train()
        avg_loss = 0.
        train_acc = 0.
        for batch_idx, (data, target) in tqdm(enumerate(self.trainset), total=len(self.trainset)):
            data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.newmodel(data)
            loss = F.cross_entropy(output, target)
            avg_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
            loss.backward()
            self.optimizer.step()

    def finetune(self):
        best_prec1 = 0
        for epoch in range(20):
            self.train()
            prec1 = self.test()
            self.scheduler.step(prec1)
            lr_current = self.optimizer.param_groups[0]['lr']
            print("currnt lr:{},current prec:{}".format(lr_current,prec1))
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best:
                ckptfile = os.path.join(self.args.savepath, 'ft_model_best.pth.tar')
            else:
                ckptfile = os.path.join(self.args.savepath, 'ft_checkpoint.pth.tar')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.newmodel.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': self.optimizer.state_dict(),
            }, ckptfile)
        return best_prec1