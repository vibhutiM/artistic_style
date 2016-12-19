import os
import sys
import scipy.io
import numpy
import tarfile

# imagenet-vgg-verydeep-19.mat

def vggLoad(filename):

    # Download the CIFAR-10 dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            
            "data",
            dataset
        )
        #f_name = new_path.replace("src/../data/%s"%dataset, "data/") 
        f_name = os.path.join(
            os.path.split(__file__)[0],
          
            "data"
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'http://www.vlfeat.org/matconvnet/models/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path) 
                     
        return new_path
    
    f_name=check_dataset(filename)

    vgg = scipy.io.loadmat(f_name)
    # f = h5py.File(f_name,'r') 

    vgg_layers =vgg['layers'][0]
    #print (vgg['meta'][0][0][2][0][0])

    types = {}
    W = {}
    b = {}

    W_NONE = ['pool','softmax']
    B_NONE = ['pool','relu','softmax']

    for layer in vgg_layers:
        layer_name = layer[0][0][0][0]
        types[layer_name]= layer[0][0][1][0]
        
        W[layer_name]= (None if types[layer_name] in W_NONE else numpy.array(layer[0][0][2][0][0].T))
        b[layer_name]= (None if types[layer_name] in B_NONE else numpy.array(layer[0][0][2][0][1].T))
        if(b[layer_name] is not None):
            b[layer_name] = b[layer_name][0]

    return W,b
    
# did numpy.array because of:
# https://groups.google.com/forum/#!topic/theano-users/tslG2ZtiwyQ  

def viewModel(vgg):
    layers = ['conv1_1','conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4','conv4_1','conv4_2','conv4_3','conv4_4','conv5_1','conv5_2','conv5_3','conv5_4']
    for layer in layers:
        print layer, type(vgg[layer]), vgg[layer].shape


if __name__ == '__main__':
    W,b=vggLoad("imagenet-vgg-verydeep-19.mat")

    viewModel(W)
