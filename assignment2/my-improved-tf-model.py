################################# Build the model ############################

# clear old variables
tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

# define model
# - 7x7 Convolutional Layer with 32 filters and stride of 1
# - ReLU Activation Layer
# - Spatial Batch Normalization Layer (trainable parameters, with scale and
#   centering)
# - 2x2 Max Pooling layer with a stride of 2
# - Affine layer with 1024 output units
# - ReLU Activation Layer
# - Affine layer from 1024 input units to 10 outputs
def my_model(X,y,is_training):
    # 7x7 Convolutional Layer with 32 filters and stride of 1
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    conv_layer = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='VALID')

    # Output shape: In each dimension we have (32-7+1)/1=26 outputs in each
    # output channel, so [N, 26, 26, 32].

    # ReLU Activation Layer
    relu_layer_1 = tf.nn.relu(conv_layer)

    # Spatial Batch Normalization Layer (trainable parameters, with scale and
    # centering)
    # 
    # Conv output is [N,W,H,F] (samples, width, height, channels), and in
    # spatial batch norm we average over N,W,H, so the features axis is 3.
    batch_norm_layer = tf.layers.batch_normalization(relu_layer_1,
                                                     axis=3,
                                                     momentum=0.99,
                                                     epsilon=0.001,
                                                     center=True,
                                                     scale=True,
                                                     training=is_training)

    # 2x2 Max Pooling layer with a stride of 2
    max_pool_layer = tf.nn.max_pool(batch_norm_layer,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    data_format='NHWC')

    # Output shape: In each dimension we have 26/2=13 due to the pooling, so
    # shape is [N, 13, 13, 32]. Total output size is 13*13*32=5408.
    total_spatial_outputs = 5408
    affine_classes_1 = 1024
    affine_classes_2 = 10

    # Affine layer with 1024 output units
    #
    # -1 means to keep the whole dimensionality constant.
    W1 = tf.get_variable("W1", [total_spatial_outputs, affine_classes_1])
    b1 = tf.get_variable("b1", [affine_classes_1])

    max_pool_flat = tf.reshape(max_pool_layer, [-1, total_spatial_outputs])
    fc_layer_1 = tf.matmul(max_pool_flat, W1) + b1

    # ReLU Activation Layer
    relu_layer_2 = tf.nn.relu(fc_layer_1)

    # Affine layer from 1024 input units to 10 outputs
    W2 = tf.get_variable("W2", [affine_classes_1, affine_classes_2])
    b2 = tf.get_variable("b2", [affine_classes_2])

    y_out = tf.matmul(relu_layer_2, W2) + b2
    return y_out

# y_out = complex_model(X,y,is_training)

######################################## Train the model #####################
