# -*- coding: utf-8 -*-
import tensorflow as tf
# =============================================================================
# 集合多个CNN网络结构用于快速训练
# - lenet
# - alexnet
# - vggnet
# - nin
# =============================================================================
class CNN(object):
    def __init__(self, num_classes=10, dropout_keep_prob=0.5):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob

    def lenet_inference(self, x):
        conv1 = self.conv(x, 5, 5, 6, 1, 1, padding='SAME', name='conv1')
        pool1 = self.max_pool(conv1, 2, 2, 2, 2, padding='VALID', name='pool1')

        conv2 = self.conv(pool1, 5, 5, 16, 1, 1, name='conv2')
        pool2 = self.max_pool(conv2, 2, 2, 2, 2, padding='VALID', name ='pool2')

        flattened=tf.contrib.layers.flatten(pool2)
        fc1 = self.fc(flattened,flattened.get_shape()[-1], 120, name='fc1')
        fc1 = self.dropout(fc1, self.dropout_keep_prob)

        fc2 = self.fc(fc1, 120, 84, name='fc2')

        self.score = self.fc(fc2, 84, self.num_classes, relu=False, name='fc3')
        return self.score
    
    def alexnet_inference(self,x):
        conv1 = self.conv(x, 11, 11, 96, 4, 4, padding='SAME', name='conv1')
        pool1 = self.max_pool(conv1, 3, 3, 2, 2, padding='SAME', name='pool1')
        norm1 = self.lrn(pool1, 2, 2e-05, 0.75, name='norm1')

        conv2 = self.conv(norm1, 5, 5, 256, 1, 1, name='conv2')
        pool2 = self.max_pool(conv2, 3, 3, 2, 2, padding='SAME', name ='pool2')
        norm2 = self.lrn(pool2, 2, 2e-05, 0.75, name='norm2')

        conv3 = self.conv(norm2, 3, 3, 384, 1, 1, name='conv3')

        conv4 = self.conv(conv3, 3, 3, 384, 1, 1,name='conv4')

        conv5 = self.conv(conv4, 3, 3, 256, 1, 1,name='conv5')
        pool5 = self.max_pool(conv5, 3, 3, 2, 2, padding='SAME', name='pool5')

        flattened=tf.contrib.layers.flatten(pool5)
        fc6 = self.fc(flattened, flattened.get_shape()[-1], 4096, name='fc6')
        fc6 = self.dropout(fc6, self.dropout_keep_prob)

        fc7 = self.fc(fc6, 4096, 4096, name='fc7')
        fc7 = self.dropout(fc7, self.dropout_keep_prob)

        self.score = self.fc(fc7, 4096, self.num_classes, relu=False, name='fc8')
        return self.score
    
    def vggnet_inference(self,x):
        conv1_1=self.conv(x,3,3,64,1,1,padding='SAME',name='conv1_1')
        conv1_2=self.conv(conv1_1,3,3,64,1,1,padding='SAME',name='conv1_2')
        pool1=self.max_pool(conv1_2,2,2,2,2,padding='SAME',name='pool1')
        
        conv2_1=self.conv(pool1,3,3,128,1,1,padding='SAME',name='conv2_1')
        conv2_2=self.conv(conv2_1,3,3,128,1,1,padding='SAME',name='conv2_2')
        pool2=self.max_pool(conv2_2,2,2,2,2,padding='SAME',name='pool2')
        
        conv3_1=self.conv(pool2,3,3,256,1,1,padding='SAME',name='conv3_1')
        conv3_2=self.conv(conv3_1,3,3,256,1,1,padding='SAME',name='conv3_2')
        conv3_3=self.conv(conv3_2,3,3,256,1,1,padding='SAME',name='conv3_3')
        pool3=self.max_pool(conv3_3,2,2,2,2,padding='SAME',name='pool3')
        
        conv4_1=self.conv(pool3,3,3,512,1,1,padding='SAME',name='conv4_1')
        conv4_2=self.conv(conv4_1,3,3,512,1,1,padding='SAME',name='conv4_2')
        conv4_3=self.conv(conv4_2,3,3,512,1,1,padding='SAME',name='conv4_3')
        pool4=self.max_pool(conv4_3,2,2,2,2,padding='SAME',name='pool4')

        conv5_1=self.conv(pool4,3,3,512,1,1,padding='SAME',name='conv5_1')
        conv5_2=self.conv(conv5_1,3,3,512,1,1,padding='SAME',name='conv5_2')
        conv5_3=self.conv(conv5_2,3,3,512,1,1,padding='SAME',name='conv5_3')
        pool5=self.max_pool(conv5_3,2,2,2,2,padding='SAME',name='pool5')

        flattened=tf.contrib.layers.flatten(pool5)
        fc6 = self.fc(flattened, flattened.get_shape()[-1], 4096, name='fc6')
        fc6 = self.dropout(fc6, self.dropout_keep_prob)
        
        fc7 = self.fc(fc6,4096, 4096, name='fc7')
        fc7 = self.dropout(fc7, self.dropout_keep_prob)
        
        self.score=self.fc(fc7,4096,self.num_classes,relu=False,name='fc8')
        return self.score
    
    def nin_inference(self,x):
        conv1_1=self.conv(x,11, 11, 96, 4, 4, padding='SAME', name='conv1_1')
        conv1_2=self.conv(conv1_1,1, 1, 96, 1, 1, name='conv1_2')
        conv1_3=self.conv(conv1_2,1, 1, 96, 1, 1, name='conv1_3')
        pool1=self.max_pool(conv1_3,3, 3, 2, 2, name='pool1')
        
        conv2_1=self.conv(pool1,5, 5, 256, 1, 1, name='conv2_1')
        conv2_2=self.conv(conv2_1,1, 1, 256, 1, 1, name='conv2_2')
        conv2_3=self.conv(conv2_2,1, 1, 256, 1, 1, name='conv2_3')
        pool2=self.max_pool(conv2_3,3, 3, 2, 2, padding='SAME', name='pool2')
        
        conv3_1=self.conv(pool2,3, 3, 384, 1, 1, name='conv3_1')
        conv3_2=self.conv(conv3_1,1, 1, 384, 1, 1, name='conv3_2')
        conv3_3=self.conv(conv3_2,1, 1, 384, 1, 1, name='conv3_3')
        pool3=self.max_pool(conv3_3,3, 3, 2, 2, padding='SAME', name='pool3')
        
        conv4_1=self.conv(pool3,3, 3, 1024, 1, 1, name='conv4_1')
        conv4_2=self.conv(conv4_1,1, 1, 1024, 1, 1, name='conv4_2')
        conv4_3=self.conv(conv4_2,1, 1, 1024, 1, 1, name='conv4_3')
        
        pool4=self.max_pool(conv4_3,6, 6, 1, 1, padding='SAME', name='pool4')
        
        flattened=tf.contrib.layers.flatten(pool4)
        self.score=self.fc(flattened, flattened.get_shape()[-1], self.num_classes,relu=False, name='fc1')
        return self.score
    
    def resnet50_inference(self,x):
        pass
    
    def inception_inference(self,x):
        pass

    def loss(self,y_predict,batch_y,is_onehot=False):
        with tf.variable_scope("loss") as scope:
            if is_onehot:
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y))
            else:
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y))
            tf.summary.scalar(scope.name+'/loss', self.loss)
        return self.loss

    def evaluation(self, y_predict, batch_y):
        with tf.variable_scope('accuracy') as scope:
            correct = tf.nn.in_top_k(y_predict, batch_y, 1)
            correct = tf.cast(correct, tf.float16)
            accuracy = tf.reduce_mean(correct)
            tf.summary.scalar(scope.name+'/accuracy', accuracy)
        return accuracy
    
    #训练
    def optimize(self, learning_rate):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
            train_op = optimizer.minimize(self.loss)
        return train_op
    

    """
    Helper methods
    """
    def conv(self,x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
        input_channels = int(x.get_shape()[-1])
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)
    
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels/groups, num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])
            
            # groups 多个GPU可以设置
            if groups == 1:
                conv = convolve(x, weights)
            else:
                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
                conv = tf.concat(axis=3, values=output_groups)
    
            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
            relu = tf.nn.relu(bias, name=scope.name)
            return relu
    
    def fc(self,x, num_in, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[num_in, num_out])
            biases = tf.get_variable('biases', [num_out])
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
            if relu == True:
                relu = tf.nn.relu(act)
                return relu
            else:
                return act
    
    def max_pool(self,x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides = [1, stride_y, stride_x, 1],
                              padding = padding, name=name)
    
    def lrn(self,x, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)
    
    def dropout(self,x, keep_prob):
        return tf.nn.dropout(x, keep_prob)
    
    



