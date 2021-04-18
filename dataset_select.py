from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import os
import glob
import numpy as np
from PIL import Image
#from scipy.misc import imread, imresize, imsave
#from scipy.misc import imresize
import imageio
from pylab import *
import json
import random
import tensorflow as tf
import cv2
#from tensorflow.contrib.slim.nets import inception

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

slim = tf.contrib.slim

FLAGS = tf.flags.FLAGS

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

#delete all of flags before running the main command     
#del_all_flags(tf.flags.FLAGS)
        


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')





tf.flags.DEFINE_string(
    'checkpoint_path_inception_v3', './models/inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v4', './models/inception_v4.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet', './models/resnet_v2_101.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_resnet_v2', './models/inception_resnet_v2_2016_08_30.ckpt', 'Path to checkpoint for inception network.')


tf.flags.DEFINE_string(
    'checkpoint_path_adv_inception_v3', './models/adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens3_adv_inception_v3', './models/ens3_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens4_adv_inception_v3', './models/ens4_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens_adv_inception_resnet_v2', './models/ens_adv_inception_resnet_v2_rename.ckpt', 'Path to checkpoint for inception network.')



tf.flags.DEFINE_string(
   'input_dir', './dataset/images', 'Input directory with images.')



tf.flags.DEFINE_string(
   'output_dir', './output', 'Output directory with images.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 50, 'How many images process at one time.')


tf.flags.DEFINE_string(
    'GPU_ID', '1,0', 'which GPU to use.')

images_num = 1000

print("print all settings\n")
print(FLAGS.master)
print(FLAGS.__dict__)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.GPU_ID


list1 = list(range(0,1000))



def load_images(input_dir, output_dir, batch_shape):
  images = np.zeros(batch_shape)
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png'))[:images_num]:
    temp_name = str.split(filepath, '/')
    output_name = output_dir + '/'+ temp_name[-1]
    # check if the file exist
    if os.path.isfile(output_name) == False:
#      with tf.gfile.Open(filepath) as f:
      with tf.io.gfile.GFile(filepath, "rb") as f:
        image = imageio.imread(f, pilmode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
      images[idx, :, :, :] = image * 2.0 - 1.0
      idx += 1
    if idx == batch_size:
      yield images
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield images



def graph(x, y):

  num_classes = 1001


  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_v3, end_points_v3 = inception_v3.inception_v3(
        x, num_classes=num_classes, is_training=False)
    
  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
        x, num_classes=num_classes, is_training=False, scope='AdvInceptionV3')

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
        x, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
        x, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

  with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
    logits_v4, end_points_v4 = inception_v4.inception_v4(
        x, num_classes=num_classes, is_training=False)

  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
        x, num_classes=num_classes, is_training=False)

  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
        x, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')

  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits_resnet, end_points_resnet = resnet_v2.resnet_v2_101(
        x, num_classes=num_classes, is_training=False)
    
    
    
    
  y_InceptionV3 = tf.argmax(end_points_v3['Predictions'], 1)
  y_InceptionV4 = tf.argmax(end_points_v4['Predictions'], 1)
  y_InceptionResnetV2 = tf.argmax(end_points_res_v2['Predictions'], 1)
  y_resnet_v2 = tf.argmax(end_points_resnet['predictions'], 1)
  y_Ens3AdvInceptionV3 = tf.argmax(end_points_ens3_adv_v3['Predictions'], 1)
  y_Ens4AdvInceptionV3 = tf.argmax(end_points_ens4_adv_v3['Predictions'], 1)
  y_EnsAdvInceptionResnetV2 = tf.argmax(end_points_ensadv_res_v2['Predictions'], 1)



  return y_InceptionV3, y_InceptionV4, y_InceptionResnetV2, y_resnet_v2, y_Ens3AdvInceptionV3, y_Ens4AdvInceptionV3, y_EnsAdvInceptionResnetV2


def main(_):

  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
    
    test_label1 = np.empty(shape=[1,0], dtype=np.int32)
    test_label2 = np.empty(shape=[1,0], dtype=np.int32)
    test_label3 = np.empty(shape=[1,0], dtype=np.int32)
    test_label4 = np.empty(shape=[1,0], dtype=np.int32)
    test_label5 = np.empty(shape=[1,0], dtype=np.int32)
    test_label6 = np.empty(shape=[1,0], dtype=np.int32)
    test_label7 = np.empty(shape=[1,0], dtype=np.int32)
    
    true_label = np.array(list1)

    pre1, pre2, pre3, pre4, pre5, pre6, pre7= graph(x_input, y)
    
    # Run computation
    s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
    s2 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
    s3 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
    s4 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
    s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
    s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
    s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
    s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))

    
    with tf.Session() as sess:
      s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
      s2.restore(sess, FLAGS.checkpoint_path_adv_inception_v3)
      s3.restore(sess, FLAGS.checkpoint_path_ens3_adv_inception_v3)
      s4.restore(sess, FLAGS.checkpoint_path_ens4_adv_inception_v3)
      s5.restore(sess, FLAGS.checkpoint_path_inception_v4)
      s6.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)
      s7.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)
      s8.restore(sess, FLAGS.checkpoint_path_resnet)
      
      b1=0
      b2=0
      b3=0
      b4=0
      b5=0
      b6=0
      b7=0
      
      fla = 0
      ima_n = 0
      images_all = glob.glob("./dataset/image_val_rename_resize/*.png")

      while (fla == 0):
        
         
          
        test_label1 = np.empty(shape=[1,0], dtype=np.int32)
        test_label2 = np.empty(shape=[1,0], dtype=np.int32)
        test_label3 = np.empty(shape=[1,0], dtype=np.int32)
        test_label4 = np.empty(shape=[1,0], dtype=np.int32)
        test_label5 = np.empty(shape=[1,0], dtype=np.int32)
        test_label6 = np.empty(shape=[1,0], dtype=np.int32)
        test_label7 = np.empty(shape=[1,0], dtype=np.int32)          
      
        for images in load_images(FLAGS.input_dir, FLAGS.output_dir, batch_shape):
          label1, label2, label3, label4, label5, label6, label7 = sess.run([pre1, pre2, pre3, pre4, pre5, pre6, pre7], feed_dict={x_input: images})
          label1 = np.add(label1, -1)
          label2 = np.add(label2, -1)
          label3 = np.add(label3, -1)
          label4 = np.add(label4, -1)
          label5 = np.add(label5, -1)
          label6 = np.add(label6, -1)
          label7 = np.add(label7, -1)
          
          test_label1 = np.append(test_label1, label1)
          test_label2 = np.append(test_label2, label2)
          test_label3 = np.append(test_label3, label3)
          test_label4 = np.append(test_label4, label4)
          test_label5 = np.append(test_label5, label5)
          test_label6 = np.append(test_label6, label6)
          test_label7 = np.append(test_label7, label7)
      
      
  
  
        a1 = (test_label1==true_label[:images_num])
        b1=a1.tolist().count(True)
        a2 = (test_label2==true_label[:images_num])
        b2=a2.tolist().count(True)
        a3 = (test_label3==true_label[:images_num])
        b3=a3.tolist().count(True)
        a4 = (test_label4==true_label[:images_num])
        b4=a4.tolist().count(True)
        a5 = (test_label5==true_label[:images_num])
        b5=a5.tolist().count(True)
        a6 = (test_label6==true_label[:images_num])
        b6=a6.tolist().count(True)
        a7 = (test_label7==true_label[:images_num])
        b7=a7.tolist().count(True)
        
        
        countn = 1000
        
        if ((b1==1000) and (b2==1000) and (b3==1000) and (b4==1000) and (b5==1000) and (b6==1000) and (b7==1000)):
        #if ((b1==1000) and (b2==1000) and (b3==1000) and (b4==1000)):
          fla = 1
        

        else: 

            
          for n in range(0,1000):
            if ((test_label1[n] != n) or (test_label2[n] != n) or (test_label3[n] != n) or (test_label4[n] != n) or (test_label5[n] != n) or (test_label6[n] != n) or (test_label7[n] != n)):
            #if ((test_label1[n] != n) or (test_label2[n] != n) or (test_label3[n] != n) or (test_label4[n] != n)):
              jpgfile = images_all[n*50 + np.random.randint(0, 50)]  
              img = cv2.imread(jpgfile)
              base = os.path.basename(jpgfile)
              a=os.path.join("./dataset/imagesnew",("00"+base[:base.find("_")])[-3:])
              cv2.imwrite(a + ".png", img)
              countn = countn -1
              
              #print (n)
            

# =============================================================================
#           jpgfile = images_all[638*50 + ima_n]
#           img = cv2.imread(jpgfile)
#           base = os.path.basename(jpgfile)
#           a=os.path.join("./dataset/imagesnew",("00"+base[:base.find("_")])[-3:])
#           cv2.imwrite(a + ".png", img)
#           ima_n = ima_n+1            
# =============================================================================
              

        print ("\n%.1f %.1f %.1f %.1f %.1f %.1f %.1f"%((b1/images_num)*100,(b2/images_num)*100,(b3/images_num)*100
                                                     ,(b4/images_num)*100,(b5/images_num)*100,(b6/images_num)*100
                                                     ,(b7/images_num)*100))
        print ("images:%d"%countn)




if __name__ == '__main__':
  tf.app.run()
