from chainer import serializers
from chainer import Variable
import numpy as np
import chainer.functions as F
from chainer import cuda
import os
import cv2
from PIL import Image
import json
import cPickle as pickle

class BaseNetwork(object):

    def __init__(self, epochs, save_every):
        self.save_every = save_every
        self.epochs=epochs
    
    def my_state(self):
        return '%s'%(self.net)
    
    def save_params(self, epoch):
        print "==> saving state %s" % self.out_model_dir
        serializers.save_hdf5("%s/net_model_classifier_%d.h5"%(self.out_model_dir, epoch),self.network)
    

    def load_state(self,path,epoch):
        print "==> loading state %s epoch %s"%(path,epoch)
        serializers.load_hdf5('./states/%s/net_model_classifier_%s.h5'%(path,epoch), self.network)
        return int(epoch)


    def read_batch(self, perm, batch_index, data_raw):

        data = np.zeros((self.batchsize, self.in_channel, self.input_height, self.input_width), dtype=np.float32)
        label = np.zeros((self.batchsize), dtype=np.int32)

        for j_,j in enumerate(perm[batch_index:batch_index+self.batchsize]):
                data[j_,:,:,:] = data_raw[j][0].astype(np.float32)
                label[j_] = int(data_raw[j][1])
        return data, label
    
    
    def step(self,perm,batch_index, mode, epoch): 
            if mode =='train':
                data, label=self.read_batch(perm,batch_index,self.train_data)
            else:
                data, label=self.read_batch(perm,batch_index,self.test_data)

            data = Variable(cuda.to_gpu(data))
            yl = self.network(data)

            label=Variable(cuda.to_gpu(label))

            L_network = F.softmax_cross_entropy(yl, label)
            A_network = F.accuracy(yl, label)

            if mode=='train':
                self.o_network.zero_grads()
                L_network.backward()
                self.o_network.update()


            return {"prediction": yl.data.get(),
                    "current_loss": L_network.data.get(),
                    "current_accuracy": A_network.data.get(),
            }

  
    def get_dataset(self, data_dir, dataset):
        if dataset=='coco':
            train_ann_path=data_dir+'/annotations/captions_train2014.json'
            test_ann_path=data_dir+'/annotations/captions_val2014.json'
            train_image_path=data_dir+'/train2014/'
            test_image_path=data_dir+'/val2014/'
            self.in_channel=3

            with open('utils/coco.pkl', 'r') as f:
                self.vocab = pickle.load(f)
            self.mydict_inv = {v:k for k, v in self.vocab.items()}

            f = open(train_ann_path,'r')
            json_data=json.load(f)
            f.close()

            train_data=[]
            test_data=[]
            self.image_hash={}

            if self.mode=='train':
                for i,caption_data in enumerate(json_data['annotations']):
                    caption_id=caption_data['id']
                    image_id=caption_data['image_id']
                    caption=caption_data['caption']

                    if image_id not in self.image_hash:
                        img=(np.asarray(Image.open(train_image_path+'COCO_train2014_%012d.jpg'%(image_id)).resize((self.input_height, self.input_width)).convert('RGB')).astype(np.int32).transpose(2, 0, 1))
                        self.image_hash[image_id]=img

                    caption = caption.replace('\n','').strip().lower()

                    if caption[-1]=='.':
                        caption=caption[0:-1]
                    caption_tokens = '<SOS> '
                    caption_tokens += caption
                    caption_tokens += ' <EOS>'
                    caption_tokens=caption_tokens.split()
                    caption_tokens=np.array([self.vocab[word] for word in caption_tokens], dtype=np.int32)

                    train_data.append((image_id,caption_tokens))

            self.in_size=len(self.vocab)

            f = open(test_ann_path,'r')
            json_data=json.load(f)
            f.close()
            image_ids=[]
            
            for i,caption_data in enumerate(json_data['annotations']):
                caption_id=caption_data['id']
                image_id=caption_data['image_id']
                caption=caption_data['caption']
                if image_id in image_ids:
                    continue  

                if image_id not in self.image_hash:
                    img=(np.asarray(Image.open(test_image_path+'COCO_val2014_%012d.jpg'%(image_id)).resize((224,224)).convert('RGB')).astype(np.int32).transpose(2, 0, 1))
                    self.image_hash[image_id]=img

                caption = caption.replace('\n','').strip().lower()

                if caption[-1]=='.':
                    caption=caption[0:-1]
                caption_tokens = caption
                caption_tokens += ' <EOS>'
                caption_tokens=caption_tokens.split()
                try:
                    caption_tokens=np.array([self.vocab[word] for word in caption_tokens], dtype=np.int32)
                except Exception:
                    continue

                
                test_data.append((image_id, caption_tokens))
                image_ids.append(image_id)


        elif dataset=='flickr8k':
            train_path=data_dir+'/Flickr_8k.trainImages.txt'
            test_path=data_dir+'/Flickr_8k.testImages.txt'
            caption_path=data_dir+'/Flickr8k.token.txt'
            data_dir=data_dir+'/Flicker8k_Dataset/'
            with open('utils/flickr.pkl', 'r') as f:
                self.vocab = pickle.load(f)
            self.mydict_inv = {v:k for k, v in self.vocab.items()}
            self.in_channel=3
            self.in_size=len(self.vocab)

            f=open(train_path,'r')
            train_images=f.read()
            f.close()
            train_images_list=train_images.split('\n')

            f=open(test_path,'r')
            test_images=f.read()
            f.close()
            test_images_list=test_images.split('\n')

            f=open(caption_path,'r')
            captions=f.read()
            f.close()   
            self.image_hash={}
            train_data=[]
            test_data=[]

            lines=captions.split('\n')     
            for i,line in enumerate(lines):
                image_id=line.split('\t')[0].split('#')[0]
                if image_id not in self.image_hash:
                    try:
                        img=(np.asarray(Image.open(data_dir+image_id).resize((self.input_height, self.input_width)).convert('RGB')).astype(np.float32).transpose(2, 0, 1))
                        self.image_hash[image_id]=img
                    except:
                        continue

                caption = line.split('\t')[1]
                caption = caption.replace('\n','').strip().lower()

                if caption[-1]=='.':
                    caption=caption[0:-1]
                caption_tokens = '<SOS> '
                caption_tokens += caption
                caption_tokens += ' <EOS>'
                caption_tokens=caption_tokens.split()
                caption_tokens=np.array([self.vocab[word] for word in caption_tokens], dtype=np.int32)
                if image_id in train_images_list:
                    train_data.append((image_id, caption_tokens))
                elif image_id in test_images_list:
                    if len(test_data)>0 and image_id == test_data[-1][0]:
                        continue
                    test_data.append((image_id, caption_tokens))


        self.out_model_dir ='./states/'+self.my_state()

        if not os.path.exists(self.out_model_dir):
            os.makedirs(self.out_model_dir)

        if self.mode=='train':
            print "==> %d training examples" % len(train_data)
            print "out_model_dir ==> %s " % self.out_model_dir
            print "==> %d test examples" % len(test_data)
        else:
            print "==> %d test examples" % len(test_data)

        return train_data, test_data
