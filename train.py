# -*- coding: utf-8 -*-
import tensorflow as tf
import input_data
import numpy as np
import argparse
import os
from AlexNet import AlexNetModel
from VggNet import VggNetModel
from RestNet import RestNetModel
from CNN import CNN
def main(_):
    train_file_path = os.path.join(FLAGS.buckets, "")
    test_file_path = os.path.join(FLAGS.buckets, "")
    logs_dir=os.path.join(FLAGS.checkpointDir, "")
    
    batch_size=32
    n_classes=2
    learning_rate=0.001
    max_step=20000
    image_size=227
    
    train_images_batch, train_labels_batch = input_data.read_and_decode(
                                    train_file_path+'train.tfrecords',batch_size,image_size)
    
    test_images_batch, test_labels_batch = input_data.read_and_decode(
                                          test_file_path+'val.tfrecords', batch_size,image_size)
    
    x = tf.placeholder(tf.float32, shape=[batch_size,image_size,image_size,3])
    y = tf.placeholder(tf.int32, shape=[batch_size,])
    keep_prob=tf.placeholder(tf.float32)
    
    model= RestNetModel(is_training=True, depth=50, num_classes=2)
    logits=model.nin_inference(x)
    loss=model.loss(logits,y,is_onehot=False)
    train_op=model.optimize(learning_rate)
    acc=model.evaluation(logits,y)
    summary_op=tf.summary.merge_all()
    
    with tf.Session() as sess:
        train_writer=tf.summary.FileWriter(logs_dir+'train/',sess.graph)
        val_writer = tf.summary.FileWriter(logs_dir+'test/', sess.graph)
        saver=tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        # start queue runner
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            for step in np.arange(max_step):
                if coord.should_stop():
                    break
                tra_images,tra_labels = sess.run([train_images_batch, train_labels_batch])
                train_loss,_=sess.run([loss,train_op],feed_dict={x:tra_images, y:tra_labels,keep_prob:0.5})
                #评估
                if step % 50==0:
                    tra_acc,summary_str = sess.run([acc,summary_op],feed_dict={x:tra_images, y:tra_labels,keep_prob:1.0})
                    train_writer.add_summary(summary_str, step)
                    
                    val_images, val_labels = sess.run([test_images_batch, test_labels_batch])
                    val_acc,summary_str = sess.run([acc,summary_op], feed_dict={x:val_images, y:val_labels,keep_prob:1.0})
                    val_writer.add_summary(summary_str, step)  
                    print('Step %d, train_loss = %.2f, train_acc = %.2f%%, val_acc = %.2f%%'
                          %(step,train_loss, tra_acc*100.0,val_acc*100.0))                               
            model_path=os.path.join(logs_dir+'model/', 'model.ckpt')
            saver.save(sess,model_path,global_step=max_step)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop() 
        # stop queue runner
        coord.request_stop()
        coord.join(threads)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    #获得buckets路径
    parser.add_argument('--buckets', type=str, default='./data/',
                        help='input data path')
    #获得checkpoint路径
    parser.add_argument('--checkpointDir', type=str, default='./logs/',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
    



