import sys, os
import numpy
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d


class ConvLayer(object):
    """ Layer of a convolution """

    def __init__(self, input, filter_shape, image_shape, f_params_w, f_params_b, subsample=(1,1), bmode =0,
            params_path = 'parameters_releasing'):
        """

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        
        """
        
        assert image_shape[1] == filter_shape[1]
        self.input = input
             
        assgn_w=np.transpose(np.load(os.path.join(params_path,f_params_w)),(3,0,1,2))
        
        self.W = theano.shared(
            np.asarray(
                assgn_w,
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        assgn_b= np.load(os.path.join(params_path,f_params_b))
        #print (assgn_b.shape)
        self.b = theano.shared(
            np.asarray(
                assgn_b,
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # convolve input feature maps with filters

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            subsample=subsample,
            border_mode=bmode,
            filter_flip=True
        )

        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        


        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

### Weights were downloaded from: https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=hdl:10864/10911


class alexNet():
    def __init__(self,params_path = 'parameters_releasing',batch_size=1, learning_rate=0.1,
                    weights=None,bias=None,image_size=(1,3,227,227)):
        

        n,d,w,h=image_size

        x = T.matrix('x')  # the data is presented as rasterized images


    



  
        self.layer0_input = x.reshape((batch_size, 3, 227, 227))
  
        self.layer0 = ConvLayer(
            input=self.layer0_input,
            image_shape=(batch_size, 3, 227, 227),
            filter_shape=(96, 3, 11, 11),
            f_params_w='W_0_65.npy',
            f_params_b='b_0_65.npy',
            subsample=(4,4),
            bmode=0,
            params_path = params_path
        )
    

    
        self.pool0=pool.pool_2d(
            input=self.layer0.output,
            ds=(3,3),
            ignore_border=True,
            st=(2,2)
        )
        
        
        self.layer1_0 = ConvLayer(
            input=self.pool0[:,:96/2,:,:],
            image_shape=(batch_size, 96/2, 27, 27),
            filter_shape=tuple(np.asarray([256/2, 96/2, 5, 5])),
            f_params_w='W0_1_65.npy',
            f_params_b='b0_1_65.npy',
            subsample=(1,1),
            bmode=2,
            params_path = params_path
        )
        
        self.layer1_1 = ConvLayer(
            input=self.pool0[:,96/2:,:,:],
            image_shape=(batch_size, 96/2, 27, 27),
            filter_shape=tuple(np.asarray([256/2, 96/2, 5, 5])),
            f_params_w='W1_1_65.npy',
            f_params_b='b1_1_65.npy',
            subsample=(1,1),
            bmode=2,
            params_path = params_path
        )
        
        self.layer1_output= T.concatenate([self.layer1_0.output, self.layer1_1.output], axis=1)
        
        self.pool1=pool.pool_2d(
            input=self.layer1_output,
            ds=(3,3),
            ignore_border=True,
            st=(2,2)
        )
        
         
        self.layer2 = ConvLayer(
            input=self.pool1,
            image_shape=(batch_size, 256, 13, 13),
            filter_shape=(384, 256, 3, 3),
            f_params_w='W_2_65.npy',
            f_params_b='b_2_65.npy',
            subsample=(1,1),
            bmode=1,
            params_path = params_path
        )
        
       
        self.layer3_0 = ConvLayer(
            input=self.layer2.output[:,:384/2,:,:],
            image_shape=(batch_size, 384/2, 13, 13),
            filter_shape=tuple(np.asarray([384/2, 384/2, 3, 3])),
            f_params_w='W0_3_65.npy',
            f_params_b='b0_3_65.npy',
            subsample=(1,1),
            bmode=1,
            params_path = params_path
        )
        
        self.layer3_1 = ConvLayer(
            input=self.layer2.output[:,384/2:,:,:],
            image_shape=(batch_size, 384/2, 13, 13),
            filter_shape=tuple(np.asarray([384/2, 384/2, 3, 3])),
            f_params_w='W1_3_65.npy',
            f_params_b='b1_3_65.npy',
            subsample=(1,1),
            bmode=1,
            params_path = params_path
        )
        
        self.layer3_output= T.concatenate([self.layer3_0.output, self.layer3_1.output], axis=1)
     
        self.layer4_0 = ConvLayer(
            input=self.layer3_output[:,:384/2,:,:],
            image_shape=(batch_size, 384/2, 13, 13),
            filter_shape=tuple(np.asarray([256/2, 384/2, 3, 3])),
            f_params_w='W0_4_65.npy',
            f_params_b='b0_4_65.npy',
            subsample=(1,1),
            bmode=1,
            params_path = params_path
        )
        
        self.layer4_1 = ConvLayer(
            input=self.layer3_output[:,384/2:,:,:],
            image_shape=(batch_size, 384/2, 13, 13),
            filter_shape=tuple(np.asarray([256/2, 384/2, 5, 5])),
            f_params_w='W1_4_65.npy',
            f_params_b='b1_4_65.npy',
            subsample=(1,1),
            bmode=1,
            params_path = params_path
        )
        
        self.layer4_output= T.concatenate([self.layer4_0.output, self.layer4_1.output], axis=1)
        
        self.pool4=pool.pool_2d(
            input=self.layer4_output,
            ds=(3,3),
            ignore_border=True,
            st=(2,2),
        )

        self.x = x
        print ("Alexnet built")

    
    

