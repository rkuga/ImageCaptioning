# ImageCaptioning

chainer implementation of Image Caption Generator

## how to use

### step1.  
install chainer
```
pip install chainer=='1.24.0'
```
  
### step2.  
download the flickr8k or mscoco dataset  

### step3.  
run the code for training  
```
python main.py --data_dir your_path --dataset flickr8k
```
  
### step4.  
after training 100 epochs, run the code for testing   
```
python main.py --mode test --data_dir your_path --dataset flickr8k --load captioning 100
```  


