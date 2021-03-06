import numpy as np
import argparse
import importlib
import sys,os
import random
from chainer import cuda


parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=128, help='size of batch')
parser.add_argument('--gpu', type=int, default=0, help='run in specific GPU')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--save_every', type=int, default=100, help='save the model every n epochs')
parser.add_argument('--net', type=str, default='captioning', help='import the network')
parser.add_argument('--load', nargs=2, type=str, default='', help='loading network parameters')
parser.add_argument('--mode', type=str, default='train', help='train/test')
parser.add_argument('--dataset', type=str, default='flickr8k', help='dataset')
parser.add_argument('--size', type=int, default=224, help='size')
parser.add_argument('--data_dir', type=str, default='.', help='dataset directory')

args = parser.parse_args()
print args

print "==> using network %s" % args.net
args_dict = dict(args._get_kwargs())

network_module = importlib.import_module("nets." + args.net)
network = network_module.Network(**args_dict)


def do_epoch(mode, epoch):
    if mode=='train':
        length=len(network.train_data)
        perm = np.random.permutation(length)
    if mode=='val':
        length=len(network.test_data)
        perm = np.array(range(length))
    sum_loss = 0
    sum_accuracy = 0
    batches_per_epoch=length//args.batchsize
    for batch_index in xrange(0, length-args.batchsize, args.batchsize):  
        step_data=network.step(perm,batch_index,mode,epoch)
        prediction = step_data["prediction"] 
        current_loss = step_data["current_loss"]
        current_accuracy = step_data["current_accuracy"]
        sum_loss += current_loss
        sum_accuracy += current_accuracy

    if mode=='train':
        print "epoch %d end loss: %.10f"%(epoch, sum_loss/batches_per_epoch),
        print "train accuracy: %.10f"%(sum_accuracy/batches_per_epoch)

    elif mode =='val':
        print "val accuracy: %.10f"%(sum_accuracy/batches_per_epoch)

start_epoch=1

if args.load != '':
    start_epoch=network.load_state(args.load[0], args.load[1])

if args.mode == 'train':
    print "==> training"  
    for epoch in xrange(start_epoch,args.epochs+1):
        do_epoch('train', epoch)
        do_epoch('val',epoch)
        if epoch % args.save_every == 0:
            network.save_params(epoch)

elif args.mode == 'test':
    print "==> testing"   
    network.test()


