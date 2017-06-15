# In[2]:


#'http://www.it.lut.fi/project/imageret/diaretdb0/'


# In[33]:

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import os
import random
from sklearn.metrics import f1_score,accuracy_score


# In[4]:


#list image files
path = '/home/sayon/diabetic_retinopathy/data/data_folder/diaretdb0_v_1_1/resources/images/'
images_list = np.sort(os.listdir(path+'diaretdb0_fundus_images/'))
labels_files = np.sort(os.listdir(path+'diaretdb0_groundtruths/'))


# In[5]:


#a = np.asarray(Image.open(path+'diaretdb0_fundus_images/'+images_list[0]))
#ima = Image.fromarray(np.uint8(a))
#print zip(images_list,labels_files)

# In[6]:

image_arrays = []
for i in images_list:
    img = Image.open(path+'diaretdb0_fundus_images/'+i)
    img_rs = img.resize((512,512), Image.ANTIALIAS)
    a = np.asarray(img_rs)
    image_arrays.append(a)
# In[7]:


image_arrays = np.asarray(image_arrays)
#image_arrays = (image_arrays - np.mean(image_arrays,axis=0))/(np.max(image_arrays,axis=0)-np.min(image_arrays,axis=0))

# In[8]:


target_labels = []
for i in labels_files:
    text_file = open(path+'diaretdb0_groundtruths/'+i, "r")
    target_labels.append(np.asarray(text_file.read().lower().split()))


# In[9]:


target_labels = np.asarray(target_labels)


# In[10]:


print target_labels.shape


# In[11]:



print image_arrays.shape


# In[12]:


target_labels_encoded = []
for i in target_labels:
    l = np.asarray([0 if j == 'n/a' else 1 for j in i])
    target_labels_encoded.append(l)


# In[13]:


target_labels_encoded = np.asarray(target_labels_encoded)


# In[14]:


target_labels_encoded.shape


# In[15]:


target_labels_modified = np.array([i/float(j) if j!=0 else i for i,j in zip(target_labels_encoded,np.sum(target_labels_encoded,axis=1))])


# In[16]:


modified_classes = np.asarray([[1,0] if i==0 else [0,1] for i in np.sum(target_labels_encoded,axis=1)])


# In[17]:


#target_labels_modified : output shape = [None,5] multi class multi output possible
#modified_classes : output shape = [None,2] normal retina and infected retina i.e. 2 classes only


# In[18]:


classes_unique = []
for i in range(target_labels.shape[1]):
    classes_unique = classes_unique + list(np.unique(target_labels[:,i])[np.unique(target_labels[:,i]) != 'n/a'])


# In[19]:


classes_unique #for target_labels_modified


# In[20]:


modified_class_labels = ['normal','infected']


# In[21]:


normal_retina_ids = np.argwhere(modified_classes[:,0]==1).reshape([-1])
affected_retina_ids = np.argwhere(modified_classes[:,0]==0).reshape([-1])


# In[22]:


#list training batches
pathn = '/home/sayon/diabetic_retinopathy/data/data_folder/diaretdb0_v_1_1/resources/'
train_batch_files = os.listdir(pathn+'traindatasets/')
test_batch_files = os.listdir(pathn+'testdatasets/')


# In[23]:


training_batches = []
for i in train_batch_files:
    text_file = open(pathn+'traindatasets/'+i, "r")
    batch_ids = [int(j)-1 for j in text_file.read().replace('image','').split('\n')  if j!='']
    training_batches.append(batch_ids)

testing_batches = []
for i in test_batch_files:
    text_file = open(pathn+'testdatasets/'+i, "r")
    batch_ids = [int(j)-1 for j in text_file.read().replace('image','').split('\n')  if j!='']
    testing_batches.append(batch_ids)    


# In[24]:


num_classes = 2 #2 if modified_classes else 5 if target_labels_modified


# In[25]:

print "presprocessing done"
#tensorflow models
tf.reset_default_graph()

def add_layer(inputs,in_shape=None,out_shape=None,layer=None,activation_fn=None,dropout=None):
    if layer=='conv':
        w = weight_variable(in_shape)
    	b = bias_variable(out_shape)
	epsilon = tf.constant(value=0.000001, shape=b.shape)
	h_conv = activation_fn(conv(inputs,w)+b+epsilon) 
        if dropout is None:
            return h_conv
        return tf.nn.dropout(h_conv,keep_prob)
    if layer=='maxpool':
        h_pool = max_pool(inputs)
        if dropout is None:
            return h_pool
        return tf.nn.dropout(h_pool,keep_prob)
    w = weight_variable(in_shape)
    b = bias_variable(out_shape)
    wxplusb = tf.matmul(inputs,w)+b
    epsilon = tf.constant(value=0.000001, shape=b.shape)
    wxplusb = wxplusb + epsilon
    outputs = activation_fn(wxplusb)
    if dropout is None:
        return outputs
    return tf.nn.dropout(outputs,keep_prob)

def weight_variable(in_shape):
    return tf.Variable(tf.truncated_normal(shape=in_shape,stddev=0.1),dtype=tf.float32)

def bias_variable(out_shape):
    return tf.Variable(tf.constant(value=0.0,shape=out_shape),dtype=tf.float32)

def conv(inputs,w):
    #strides [1,x_movement,y_movement,1]
    #stride[0] = stride[3] = 1
    return tf.nn.conv2d(inputs,w,strides=[1,1,1,1],padding='SAME')

def max_pool(x):
    # strides [1,x_movement,y_movement,1]
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')


# In[26]:


xs = tf.placeholder(dtype=tf.float32,shape=[None,512,512,3])
ys = tf.placeholder(dtype=tf.float32,shape=[None,num_classes])
keep_prob = tf.placeholder(tf.float32)


# In[27]:


conv_layer_1 =  add_layer(xs,[3,3,3,32],[32],activation_fn=tf.nn.relu,layer='conv')
#epsilon = tf.constant(value=0.00001, shape=conv_layer_1.shape)
#conv_layer_1 = conv_layer_1 + epsilon
#3x3 strides 32 filters, RELU
#patch 3x3, in size 3, out size 32
max_pool_1 = add_layer(conv_layer_1,layer='maxpool')
conv_layer_2 =  add_layer(max_pool_1,[3,3,32,64],[64],activation_fn=tf.nn.relu,layer='conv')
#epsilon = tf.constant(value=0.00001, shape=conv_layer_2.shape)
#conv_layer_2 = conv_layer_2 + epsilon
#3x3 strides 32 filters, RELU
#patch 3x3, in size 32, out size 32
max_pool_2 = add_layer(conv_layer_2,layer='maxpool')
#conv_layer_3 = add_layer(max_pool_2,[3,3,64,128],[128],activation_fn=tf.nn.relu,layer='conv') 
#3x3 strides 64 filters, RELU
#patch 3x3, in size 32, out size 64
#max_pool_3 = add_layer(conv_layer_3,[3,3,128,128],[128],layer='maxpool')
#conv_layer_4 = add_layer(max_pool_3,[3,3,64,64],[64],activation_fn=tf.nn.,layer='conv') 
#3x3 strides 64 filters, RELU
#patch 3x3, in size 64, out size 64
#max_pool_4 = add_layer(conv_layer_4,[3,3,64,64],[64],layer='maxpool')
#conv_layer_5 =  add_layer(max_pool_4,[3,3,64,128],[128],activation_fn=tf.nn.sigmo,layer='conv')
#3x3 strides 128 filters, RELU
#patch 3x3, in size 64, out size 128
#max_pool_5 = add_layer(conv_layer_5,[3,3,128,128],[128],activation_fn=tf.nn.relu,layer='maxpool')
#conv_layer_6 = add_layer(max_pool_5,[3,3,128,128],[128],activation_fn=tf.nn.relu,layer='conv')
#3x3 strides 128 filters, RELU
#patch 3x3, in size 128, out size 128
#max_pool_6 = add_layer(conv_layer_6,[3,3,128,128],[128],activation_fn=tf.nn.relu,layer='maxpool')
#conv_layer_7 = add_layer(max_pool_6,[3,3,128,256],[256],activation_fn=tf.nn.relu,layer='conv')
#3x3 strides 256 filters, RELU
#patch 3x3, in size 128, out size 256
conv_layer_8 = add_layer(max_pool_2,[3,3,64,128],[128],activation_fn=tf.nn.relu,layer='conv')
#3x3 strides 256 filters, RELU
#patch 3x3, in size 256, out size 256
#max_pool_7 = add_layer(conv_layer_8,[3,3,256,256],[256],activation_fn=tf.nn.relu,layer='maxpool')
max_pool_3 = add_layer(conv_layer_8,layer='maxpool')
conv_layer_9 = add_layer(max_pool_3,[3,3,128,256],[256],activation_fn=tf.nn.relu,layer='conv')
#3x3 strides 512 filters, RELU 
#patch 3x3, in size 256, out size 512
max_pool_8 = add_layer(conv_layer_9,layer='maxpool')
#conv_layer_10 = add_layer(max_pool_8,[3,3,512,512],[512],activation_fn=tf.nn.relu,layer='conv')
#3x3 strides 512 filters, RELU
#patch 3x3, in size 512, out size 512
conv_layer_10 = add_layer(max_pool_8,[3,3,256,512],[512],activation_fn=tf.nn.relu,layer='conv')
max_pool_9 = add_layer(conv_layer_10,layer='maxpool')
conv_layer_11 = add_layer(max_pool_9,[3,3,512,512],[512],activation_fn=tf.nn.relu,layer='conv')
max_pool_10 = add_layer(conv_layer_11,layer='maxpool')
#max_pool_11 = add_layer(max_pool_10,layer='maxpool')
#dropout 0.5
#print max_pool_10.shape
#sys.exit()
flat_1 = tf.reshape(max_pool_10,[-1,8*8*512])
fc_1 = add_layer(flat_1,[8*8*512,1024],[1024],activation_fn=tf.nn.relu)
#1024 as out_shape, RELU, dropout 0.5
fc_2 = add_layer(fc_1,[1024,1024],[1024],activation_fn=tf.nn.relu)
#1024 as out_shape, RELU
pred = add_layer(fc_2,[1024,num_classes],[num_classes],activation_fn=tf.nn.softmax)
#num_classes as out_shape with softmax 
#print pred.shape
#sys.exit()
# In[28]:


#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(tf.clip_by_value(pred,1e-10,1.0)),reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(pred-ys),axis=1))

# In[29]:


training = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)


# In[30]:


init = tf.global_variables_initializer()
print "tf graph made"

# In[46]:
#print normal_retina_ids
#print affected_retina_ids

nrs = random.sample(normal_retina_ids,15)
ars = random.sample(affected_retina_ids,85)
training_sample_ids = nrs + ars
testing_sample_ids = list(set(range(modified_classes.shape[0]))-set(training_sample_ids))
test_x = image_arrays[testing_sample_ids]
test_y = modified_classes[testing_sample_ids]


# In[47]:


epochs = 300
batch_size = 100 #50 from nrs and 50 from ars


# In[1]:
print "session starts"

with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        training_batch_ids = random.sample(nrs,10)+random.sample(nrs,10) +random.sample(nrs,10) +random.sample(nrs,10) +random.sample(nrs,10)  + random.sample(ars,50)
	random.shuffle(training_batch_ids)
        batch_x = image_arrays[training_batch_ids]
	# print batch_x
        batch_y = modified_classes[training_batch_ids]
        #print batch_y
	#break
	_,loss = sess.run([training,cross_entropy],feed_dict={xs:batch_x,ys:batch_y,keep_prob:0.5})
        #training_fscore = f1_score(np.argmax(batch_y,axis=1),np.argmax(ypred,axis=1))
        #training_acc = accuracy_score(np.argmax(batch_y,axis=1),np.argmax(ypred,axis=1))
        print "step: {1} > training- loss: {0}".format(loss,i+1)
        if i>200:
            ypred = sess.run(pred,feed_dict={xs:test_x,keep_prob:1})
            testing_fscore = f1_score(np.argmax(test_y,axis=1),np.argmax(ypred,axis=1))
            testing_acc = accuracy_score(np.argmax(test_y,axis=1),np.argmax(ypred,axis=1))
            print "step: {2} > training- f1_score: {0},accuracy: {1}".format(testing_fscore,testing_acc,i+1)



