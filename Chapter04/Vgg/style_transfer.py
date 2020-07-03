#Import the libraries needed
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf 
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


#formatting
##from IPython.core.display import HTML
##HTML("""<style> .rendered_html code { 
##    padding: 2px 4px;
##    color: #c7254e;
##    background-color: #f9f2f4;
##    border-radius: 4px;`
##} </style>""")

#load the vgg-19 layer model
vgg = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
vgg_layers = vgg['layers']
vgg_layers.shape

[print (vgg_layers[0][i][0][0][0][0]) for i in range(43)]


[print (vgg_layers[0][i][0][0][2][0][0].shape,\
        vgg_layers[0][i][0][0][0][0]) for i in range(43) 
 if 'conv' in vgg_layers[0][i][0][0][0][0] \
 or 'fc' in vgg_layers[0][i][0][0][0][0]]


def weights(layer, expected_layer_name):
    """
    Return the kernels/weights and bias from the VGG model for a given layer.
    """
    W = vgg_layers[0][layer][0][0][2][0][0]
    b = vgg_layers[0][layer][0][0][2][0][1]
    layer_name = vgg_layers[0][layer][0][0][0][0]
    
    #to check we obtained the correct layer from the vgg model
    assert layer_name == expected_layer_name
    return W, b

def relu(conv2d_layer):
    """
    Return the RELU activation applied onto a conv2d layer
    """
    return tf.nn.relu(conv2d_layer)

def conv2d(prev_layer, layer, layer_name):
    """
    Return the Conv2D layer using the weights, biases from the VGG
    model at 'layer'.
    """
    W, b = weights(layer, layer_name)
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    return tf.nn.conv2d(
        prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

def conv2d_relu(prev_layer, layer, layer_name):
    """
    Return the Conv2D + RELU layer using the weights, biases from the VGG
    model at 'layer'.
    """
    return relu(conv2d(prev_layer, layer, layer_name))

def avgpool(prev_layer):
    """
    Return the AveragePooling layer.
    """
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#Image width and height of the content and style image, they don't need to be equal
IMAGE_WIDTH = 478
IMAGE_HEIGHT = 478
INPUT_CHANNELS = 3 #RGB image

# Connect the layers to create the model
model = {}
model['input'] = tf.Variable(np.zeros((1, IMAGE_HEIGHT,IMAGE_WIDTH,INPUT_CHANNELS)),dtype = 'float32')


model['conv1_1']  = conv2d_relu(model['input'], 0, 'conv1_1')
model['conv1_2']  = conv2d_relu(model['conv1_1'], 2, 'conv1_2')
model['avgpool1'] = avgpool(model['conv1_2'])
model['conv2_1']  = conv2d_relu(model['avgpool1'], 5, 'conv2_1')
model['conv2_2']  = conv2d_relu(model['conv2_1'], 7, 'conv2_2')
model['avgpool2'] = avgpool(model['conv2_2'])
model['conv3_1']  = conv2d_relu(model['avgpool2'], 10, 'conv3_1')
model['conv3_2']  = conv2d_relu(model['conv3_1'], 12, 'conv3_2')
model['conv3_3']  = conv2d_relu(model['conv3_2'], 14, 'conv3_3')
model['conv3_4']  = conv2d_relu(model['conv3_3'], 16, 'conv3_4')
model['avgpool3'] = avgpool(model['conv3_4'])
model['conv4_1']  = conv2d_relu(model['avgpool3'], 19, 'conv4_1')
model['conv4_2']  = conv2d_relu(model['conv4_1'], 21, 'conv4_2')
model['conv4_3']  = conv2d_relu(model['conv4_2'], 23, 'conv4_3')
model['conv4_4']  = conv2d_relu(model['conv4_3'], 25, 'conv4_4')
model['avgpool4'] = avgpool(model['conv4_4'])
model['conv5_1']  = conv2d_relu(model['avgpool4'], 28, 'conv5_1')
model['conv5_2']  = conv2d_relu(model['conv5_1'], 30, 'conv5_2')
model['conv5_3']  = conv2d_relu(model['conv5_2'], 32, 'conv5_3')
model['conv5_4']  = conv2d_relu(model['conv5_3'], 34, 'conv5_4')
model['avgpool5'] = avgpool(model['conv5_4'])


MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

def preprocess(path):
    """ Loads the path and then preprocess so the image is
        fit for inputting into the network"""
                    
    image = plt.imread(path)
    # The input requires an extra dimension for the number of images
    image = image[np.newaxis]
    # Input to the VGG model expects the mean to be subtracted.
    image = image - MEAN_VALUES
    return image

def deprocess(path, image):
    """ Does the opposite of preprocess so that
        what remains may be saved as an image"""
    #add the mean back for the correct colours
    image = image + MEAN_VALUES
    # Remove the 1st dimension that was added in preprocess
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

    
#IMAGE A
import PIL
from PIL import Image
content_image = Image.open('cat.jpg')
content_image = content_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), PIL.Image.ANTIALIAS)
content_image.save('cat.jpg')
content_image = preprocess('cat.jpg')


#IMAGE B
plt.figure(figsize=(12,24))
plt.subplot(1,2,1)
plt.title('Mean subtracted')
imshow(content_image[0])
plt.subplot(1,2,2)
imshow((content_image + MEAN_VALUES)[0].astype('uint8'))
style_image = preprocess('mosaic.jpg') 



plt.figure(figsize=(12,24))
plt.subplot(1,2,1)
plt.title('Mean subtracted')
imshow(style_image[0])
plt.subplot(1,2,2)
imshow((style_image + MEAN_VALUES)[0].astype('uint8'))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(content_image))
outputs = [sess.run(model['conv%d_2'%i]) for i in range(1,6)]
[['conv%d_2'%(i+1), o.shape] for i,o in enumerate(outputs)]



def contentloss(p, x):
    size = np.prod(p.shape[1:]) 
    loss = (1./(2*size)) * tf.reduce_sum(tf.pow((x - p),2)) 
    return loss


noise_image = np.random.uniform(-20, 20,(1, IMAGE_HEIGHT,IMAGE_WIDTH,INPUT_CHANNELS)).astype('float32')

#the effect of this parameter plays a role in how much of the style will be embedded into the content image
noise_content_ratio = 0.7
input_noise = noise_image * noise_content_ratio + content_image * (1 - noise_content_ratio)
plt.figure(figsize=(10,20))
plt.subplot(1,2,1)
plt.title('input')
imshow(input_noise[0])
plt.subplot(1,2,2)
plt.title('input without the mean subtracted')
imshow((input_noise + MEAN_VALUES)[0].astype('uint8'))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# Construct content_loss using content_image and noise_input (assigned later).
sess.run(model['input'].assign(content_image))
#content_loss = contentloss(sess.run(model['conv5_3']), model['conv5_3'])

content_loss = contentloss(sess.run(model['conv5_4']), model['conv5_4'])

# Construct style_loss using style_image.

#assign the style image as input
sess.run(model['input'].assign(style_image))

#layers and weights in equation 5
STYLE_LAYERS = [
    ('conv1_1', 0.5),
    ('conv2_1', 1.3),
    ('conv3_1', 2.3),
    ('conv4_1', 3.1),
    ('conv5_1', 4.0),]

def gram_matrix(F, N, M):
    """
    The gram matrix G.
    """
    Ft = tf.reshape(F, (M, N))
    return tf.matmul(tf.transpose(Ft), Ft)

def style_loss(a, x):
    N = a.shape[3]
    M = a.shape[1] * a.shape[2]
    A = gram_matrix(a, N, M)
    # G is the style representation of the input_noise image in the same layer.
    G = gram_matrix(x, N, M)
    result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
    return result

#sess.run(model[layer_name]) will be the output of the layer when the noise image is assigned as input later
#model[layer_name] will be the output of the layer when the style iamge is assigned as input
E = [style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in STYLE_LAYERS]
W = [w for _, w in STYLE_LAYERS]

#equation 5 in the paper
styleloss = sum([W[l] * E[l] for l in range(len(STYLE_LAYERS))])

alpha = 1
beta = 100

#equation 7 from the paper
total_loss = alpha * content_loss + beta * styleloss

#minimize the total loss
train_step = tf.train.AdamOptimizer(1.5).minimize(total_loss)

sess.run(tf.global_variables_initializer())

#assign the noise image as input
sess.run(model['input'].assign(input_noise))

for it in range(2001):
    sess.run(train_step)
    #obtain the trained input_noise image every 100 iterations
    if it%100 == 0:
        mixed_image = sess.run(model['input'])
        print('iteration:',it,'cost: ', sess.run(total_loss))
        filename = 'out/%d.png' % (it)
        deprocess(filename, mixed_image)

"""
x=np.linspace(0,2000,21)
y=np.array([745494000000.0,
 8243060000.0,
 2974590000.0,
 1756650000.0,
 1238110000.0,
 943114000.0,
 750433000.0,
 615972000.0,
 517077000.0,
 441694000.0,
 382687000.0,
 335401000.0,
 296861000.0,
 264709000.0,
 237690000.0,
 214629000.0,
 194709000.0,
 177407000.0,
 162231000.0,
 148853000.0,
 137001000.0])
plt.style.use('ggplot')
plt.plot(x,np.log(y))
plt.ylabel('log(cost)', rotation=0)
plt.xlabel('iteration')
plt.xticks(np.linspace(0,2000,21), rotation=45);
plt.yticks([np.log(137001000.0),np.log(214629000.0),np.log(382687000.0),np.log(943114000.0),np.log(745494000000.0)]);
"""
plt.figure(figsize=(15,15))
for i in range(21):

    plt.subplot(5,5,i+1)
    imshow(plt.imread('out/%d00.png' % i))
    plt.axis('off')
    plt.title('iteration %d00' % i)




