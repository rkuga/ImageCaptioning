import numpy as np
import random
import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
import chainer.functions as F
import chainer.links as L
from base_network import BaseNetwork
from chainer.links.model.vision.googlenet import prepare as google_prepare
from utils.beam_search import beam_search
from progressbar import ProgressBar

class GoogLeNet(chainer.Chain):
    def __init__(self):
        w = chainer.initializers.HeNormal()
        super(GoogLeNet, self).__init__(
            google = L.GoogLeNet(),
            )

    def __call__(self, x, train=True, test=False):
        h = self.google(x,layers=['pool5'])

        return h['pool5']

class Decoder(chainer.Chain):
    def __init__(self, in_size):
        super(Decoder, self).__init__(     
            # decoder
            embed=L.EmbedID(in_size, 1024,ignore_label=-1),
            l1_x=L.Linear(1024, 4*1024),
            l1_h=L.Linear(1024, 4*1024),
            out=L.Linear(1024, in_size),

        )

    def __call__(self,feature, state, test=False, train=True, image=False):
        if image:
            h1_in = self.l1_x(feature) + self.l1_h(state['h1'])
            c1, h1 = F.lstm(state['c1'], h1_in)
            y = self.out(h1)
            state = {'c1': c1, 'h1': h1}            
        else:
            h0 = self.embed(feature)
            h1_in = self.l1_x(h0) + self.l1_h(F.dropout(state['h1'], train=train))
            c1, h1 = F.lstm(state['c1'], h1_in)
            y = self.out(h1)
            state = {'c1': c1, 'h1': h1}
        return state, y



class Network(BaseNetwork):
    def __init__(self,gpu,batchsize,data_dir,dataset,net,mode,epochs,save_every,size,**kwargs):
        super(Network, self).__init__(epochs,save_every)
        print "building ..."
        self.input_height=size
        self.input_width=size
        self.net = net
        self.mode=mode
        self.dataset=dataset
        self.train_data, self.test_data=self.get_dataset(data_dir,dataset)
        print 'input_channel ==> %d using %s dataset'%(self.in_channel, self.dataset)

        self.enc = GoogLeNet()
        self.dec = Decoder(self.in_size)

        self.xp = cuda.cupy
        cuda.get_device(gpu).use()

        self.enc.to_gpu()
        self.dec.to_gpu()

        self.o_dec = optimizers.RMSpropGraves()
        self.o_dec.setup(self.dec)

        self.batchsize=batchsize

    def my_state(self):
        return '%s'%(self.net)

    def read_batch(self, perm, batch_index,data_raw,mode):
        max_length=0
        for j in perm[batch_index:batch_index+self.batchsize]:
            if len(data_raw[j][1]) > max_length:
                max_length = len(data_raw[j][1])

        data = np.zeros((self.batchsize, self.in_channel, 224, 224), dtype=np.float32)
        label=np.zeros((self.batchsize,max_length), dtype=np.int32)-1
        first_words=np.zeros((self.batchsize), dtype=np.int32)-1

        for j_,j in enumerate(perm[batch_index:batch_index+self.batchsize]):

            image = (self.image_hash[data_raw[j][0]])
            image = google_prepare(image)
            data[j_,:,:,:] = image
            first_words[j_]=data_raw[j][1][0]
            if len(data_raw[j][1]) < max_length:
                arr = np.concatenate((data_raw[j][1], np.zeros((max_length-len(data_raw[j][1])))-1))
                label[j_]=arr
            else:
                label[j_]=data_raw[j][1]
 
        return data, first_words, label


    def step(self,perm,batch_index,mode,epoch): 
        if mode=='train':
            data, first_words, label=self.read_batch(perm,batch_index,self.train_data,mode)
            train = True
        else :
            data, first_words, label=self.read_batch(perm,batch_index,self.test_data,mode)
            train = False

        data = Variable(cuda.to_gpu(data))
        state = {name: Variable(self.xp.zeros((self.batchsize, 1024),dtype=self.xp.float32)) for name in ('c1', 'h1')}
        loss=Variable(cuda.cupy.asarray(0.0).astype(np.float32))
        acc=0.0

        ### image-encoder ###
        h = self.enc(data, train=train, test=not train)
        h=h.data
        h=Variable(h)


        ### first LSTM ###
        state,_ = self.dec(h, state,train=train, test=not train, image=True)
        ### input <SOS> ###
        state,y = self.dec(Variable(cuda.to_gpu(first_words)), state,train=train, test=not train)
        loss += F.softmax_cross_entropy(y, Variable(cuda.to_gpu(label.T[1])))
        acc += F.accuracy(y, Variable(cuda.to_gpu(label.T[1])), ignore_label=-1).data.get()

        for cur_word,next_word in zip(label.T[1:-1],label.T[2:]):
            state,y = self.dec(Variable(cuda.to_gpu(cur_word)), state,train=train, test=not train)
            loss += F.softmax_cross_entropy(y, Variable(cuda.to_gpu(next_word)))
            acc += F.accuracy(y, Variable(cuda.to_gpu(next_word)), ignore_label=-1).data.get()

        if mode=='train':
            self.dec.cleargrads()    
            loss.backward()
            self.o_dec.update()


        return {"prediction": 0,
                "current_loss": loss.data.get()/(label.T.shape[0]),
                "current_accuracy": acc/(label.T.shape[0]),
                }


    def test(self):
        p = ProgressBar()
        f = open('./captions.txt', 'w')
        for i_  in p(range(0,len(self.test_data),self.batchsize)): 
            data = np.zeros((self.batchsize, self.in_channel, self.input_height, self.input_width), dtype=np.float32)
            t2 = np.zeros((self.batchsize, self.input_height, self.input_width), dtype=np.int32)
            label=[]
            first_words=np.zeros((self.batchsize), dtype=np.int32)
            for j in xrange(self.batchsize):
                image = (self.image_hash[self.test_data[i_+j][0]])
                image = google_prepare(image)
                data[j,:,:,:] = image
                label.append(self.test_data[i_+j][1])
                first_words[j]=self.test_data[i_+j][1][0]

            genrated_sentence=[]
            data = Variable(cuda.to_gpu(data))
            state = {name: Variable(self.xp.zeros((data.shape[0], 1024),dtype=self.xp.float32)) for name in ('c1', 'h1')}
            h = self.enc(data, train=False, test=True)
            
            ### first LSTM ###
            state,_ = self.dec(h, state,train=False, test=True, image=True)
            ### input <SOS> ###
            state,y = self.dec(Variable(cuda.to_gpu(first_words)), state,train=False, test=True)

            genrated_sentence_beamed = beam_search(self.dec,state,y,data, 20, self.mydict_inv)
            
            # maximum sentence length is 50
            for i in xrange(50):
                y = Variable(self.xp.array(np.argmax(y.data.get(), axis=1)).astype(self.xp.int32))
                state,y = self.dec(y, state,train=False, test=True)
                genrated_sentence.append(y.data)

            
            for b in range(self.batchsize):
                f.write(str(self.test_data[i_+b][0])+'/')
                # GT caption
                for i in range(1,len(label[b])-1):
                    index=label[b][i]
                    f.write(self.mydict_inv[index]+' ')
                f.write("/")

                # Predicted caption
                for i,predicted_word in enumerate(genrated_sentence):
                    index=cuda.to_cpu(predicted_word.argmax(1))[b]
                    if self.mydict_inv[index]=='<EOS>':
                        break
                    f.write(self.mydict_inv[index]+' ')
                f.write("/")

                # beamed caption
                for i in range(len(genrated_sentence_beamed[b])):
                    index=genrated_sentence_beamed[b][i]
                    if self.mydict_inv[index]=='<EOS>':
                        break
                    f.write(self.mydict_inv[index]+' ')
                f.write("\n")

        f.close()


    def save_params(self, epoch):
        print "==> saving state %s" % self.out_model_dir
        serializers.save_hdf5("%s/net_model_enc_%d.h5"%(self.out_model_dir, epoch),self.enc)
        serializers.save_hdf5("%s/net_model_dec_%d.h5"%(self.out_model_dir, epoch),self.dec)


    def load_state(self,path,epoch):
        print "==> loading state %s epoch %s"%(path,epoch)
        serializers.load_hdf5('./states/%s/net_model_enc_%s.h5'%(path,epoch), self.enc)
        serializers.load_hdf5('./states/%s/net_model_dec_%s.h5'%(path,epoch), self.dec)

        return int(epoch)

