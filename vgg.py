import sys, os
import numpy
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d


class LeNetConvLayer(object):
    """Convolutional Layer"""

    def __init__(self,input, filter_shape, image_shape,weight=None,bias=None):
        
        
        self.input = input

        
        W = theano.shared( weight, borrow=True)

        b = theano.shared(value=bias, borrow=True)

        self.W = W
        self.b = b
        
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode='half'
        )

        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        


class VGG_19():
    
    def __init__(self, batch_size=1, weights=None,bias=None,image_size=(1,3,224,224)):
        

        batch,c,width,height=image_size

        x = T.matrix('x')  
        
        layer0_input = x.reshape((batch_size, 3, width, height))

        self.conv1_1 = LeNetConvLayer(
            input=layer0_input,
            image_shape=(batch_size, 3, width, height),
            filter_shape=(64, 3, 3, 3),
            weight=weights['conv1_1'],
            bias=bias['conv1_1']
        )
        
        
        self.conv1_2 = LeNetConvLayer(
            input=self.conv1_1.output,
            image_shape=(batch_size, 64, width, height),
            filter_shape=(64, 64, 3, 3),
            weight=weights['conv1_2'],
            bias=bias['conv1_2']
        )
        
        pool1_output = pool.pool_2d(
                input=self.conv1_2.output,
                ds=(2,2),
                ignore_border=True,
            )

        width = width/2
        height = height/2

        self.conv2_1 = LeNetConvLayer(
            input=pool1_output,
            image_shape=(batch_size, 64, width, height),
            filter_shape=(128, 64, 3, 3),
            weight=weights['conv2_1'],
            bias=bias['conv2_1']
        )
        

        self.conv2_2 = LeNetConvLayer(
            input=self.conv2_1.output,
            image_shape=(batch_size, 128, width, height),
            filter_shape=(128, 128, 3, 3),
            weight=weights['conv2_2'],
            bias=bias['conv2_2']
        )
        

        pool2_output   = pool.pool_2d(
                input=self.conv2_2.output,
                ds=(2,2),
                ignore_border=True,
            )

        width = width/2
        height = height/2

        self.conv3_1 = LeNetConvLayer(
            input=pool2_output,
            image_shape=(batch_size, 128, width, height),
            filter_shape=(256, 128, 3, 3),
            weight=weights['conv3_1'],
            bias=bias['conv3_1']
        )
       

        self.conv3_2 = LeNetConvLayer(
            input=self.conv3_1.output,
            image_shape=(batch_size, 256, width, height),
            filter_shape=(256, 256, 3, 3),
            weight=weights['conv3_2'],
            bias=bias['conv3_2']
        )
       

        self.conv3_3 = LeNetConvLayer(
            input=self.conv3_2.output,
            image_shape=(batch_size, 256, width, height),
            filter_shape=(256, 256, 3, 3),
            weight=weights['conv3_3'],
            bias=bias['conv3_3']
        )
        

        self.conv3_4 = LeNetConvLayer(
            input=self.conv3_3.output,
            image_shape=(batch_size, 256, width, height),
            filter_shape=(256, 256, 3, 3),
            weight=weights['conv3_4'],
            bias=bias['conv3_4']
        )


        pool3_output   = pool.pool_2d(
                input=self.conv3_4.output,
                ds=(2,2),
                ignore_border=True,
            )

        width = width/2
        height = height/2

        self.conv4_1 = LeNetConvLayer(
            input=pool3_output,
            image_shape=(batch_size, 256, width, height),
            filter_shape=(512, 256, 3, 3),
            weight=weights['conv4_1'],
            bias=bias['conv4_1']
        )
        
        self.conv4_2 = LeNetConvLayer(
            input=self.conv4_1.output,
            image_shape=(batch_size, 512, width, height),
            filter_shape=(512, 512, 3, 3),
            weight=weights['conv4_2'],
            bias=bias['conv4_2']
        )
        

        self.conv4_3 = LeNetConvLayer(
            input=self.conv4_2.output,
            image_shape=(batch_size, 512, width, height),
            filter_shape=(512, 512, 3, 3),
            weight=weights['conv4_3'],
            bias=bias['conv4_3']
        )



        self.conv4_4 = LeNetConvLayer(
            input=self.conv4_3.output,
            image_shape=(batch_size, 512, width, height),
            filter_shape=(512, 512, 3, 3),
            weight=weights['conv4_4'],
            bias=bias['conv4_4']
        )
        
        pool4_output   = pool.pool_2d(
                input=self.conv4_4.output,
                ds=(2,2),
                ignore_border=True,
            )

        width = width/2
        height = height/2

        self.conv5_1 = LeNetConvLayer(
            input=pool4_output,
            image_shape=(batch_size, 512, width, height),
            filter_shape=(512, 512, 3, 3),
            weight=weights['conv5_1'],
            bias=bias['conv5_1']
        )
        

        self.conv5_2 = LeNetConvLayer(
            input=self.conv5_1.output,
            image_shape=(batch_size, 512, width, height),
            filter_shape=(512, 512, 3, 3),
            weight=weights['conv5_2'],
            bias=bias['conv5_2']
        )
        

        self.conv5_3 = LeNetConvLayer(
            input=self.conv5_2.output,
            image_shape=(batch_size, 512, width, height),
            filter_shape=(512, 512, 3, 3),
            weight=weights['conv5_3'],
            bias=bias['conv5_3']
        )
        
        self.conv5_4 = LeNetConvLayer(
            input=self.conv5_3.output,
            image_shape=(batch_size, 512, width, height),
            filter_shape=(512, 512, 3, 3),
            weight=weights['conv5_4'],
            bias=bias['conv5_4']
        )
        

        pool5_output   = pool.pool_2d(
                input=self.conv5_4.output,
                ds=(2,2),
                ignore_border=True,
            )

        self.x = x
        print('Weights Loaded in VGG')
