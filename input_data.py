# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from PIL import Image

def get_files(file_dir):
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))
    
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    
    # 分离一部分作为测试集
    number=round(len(image_list)*0.8)
    print("train_set length is:%d" % number)
    train_image_list=image_list[:number]
    val_image_list=image_list[number:]
    
    train_label_list=label_list[:number]
    val_label_list=label_list[number:]
    return train_image_list, train_label_list, val_image_list,val_label_list


def int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def write_to_record(image_list,label_list,image_size,output_data):
    #判断文件是否存在
    if os.path.exists(output_data):
        print("record has already existed")
    else:
        
        n_samples = len(label_list)
        
        if np.shape(image_list)[0] != n_samples:
            raise ValueError('Images size %d does not match label size %d.' %(image_list.shape[0], n_samples))
        
        # wait some time here, transforming need some time based on the size of your data.
        writer = tf.python_io.TFRecordWriter(output_data)
        print('\nTransform start......')
        for i in np.arange(0, n_samples):
            try:
                image = Image.open(image_list[i]) # type(image) must be array!]
                image = image.resize((image_size,image_size))
                image_raw = image.tobytes()
                label = int(label_list[i])
                example = tf.train.Example(features=tf.train.Features(feature={
                                'label':int64_feature(label),
                                'image_raw': bytes_feature(image_raw)}))
                writer.write(example.SerializeToString())
            except IOError as e:
                print('Could not read:', image_list[i])
                print('error: %s' %e)
                print('Skip it!\n')
        writer.close()
        print('Transform done!')

def read_and_decode(tfrecords_file, batch_size,image_size):

    # make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
                                        serialized_example,
                                        features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    
    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.
    image = tf.reshape(image, [image_size, image_size,3])
    image = tf.image.per_image_standardization(image)
    label = tf.cast(img_features['label'], tf.int32)    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = 3*batch_size)
    return image_batch, tf.reshape(label_batch, [batch_size])
#%%
# To test the generated batches of images
# When training the model, DO comment the following codes

if __name__=='__main__':
    import matplotlib.pyplot as plt
    def plot_images(images, labels):
        for i in np.arange(0, 25):
            plt.subplot(5, 5, i + 1)
            plt.axis('off')
            plt.title(labels[i], fontsize = 14)
            plt.subplots_adjust(top=1.5)
            plt.imshow(images[i])
        plt.show()
    batch_size = 32
    image_size=227
    
    #data_dir="/Users/yangzhan/MachineLearning/data/DogsvsCats/train/" #macos
    data_dir=r'D:\MachineLearning\data\DogsvsCats\train\\' #windows
    train_data = "./data/train.tfrecords"
    val_data="./data/val.tfrecords"

    #写tfrecord
    train_image_list,train_label_list,val_image_list,val_label_list=get_files(data_dir)
    write_to_record(train_image_list,train_label_list,image_size=image_size,output_data=train_data)
    write_to_record(val_image_list,val_label_list,image_size=image_size,output_data=val_data)
    #%%
    #读tfrecord
    image_batch, label_batch = read_and_decode(train_data, batch_size=batch_size,image_size=image_size)
    with tf.Session()  as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        try:
            while not coord.should_stop() and i<1:
                # just plot one batch size            
                image, label = sess.run([image_batch, label_batch])
                plot_images(image, label)
                i+=1
                
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)