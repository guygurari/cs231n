from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_flat = x.reshape(N,D)
    out = x_flat.dot(w) + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M) 
            (derivative of loss wrt output of this layer)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N = x.shape[0]
    D = np.prod(x.shape[1:])

    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(N,D).T.dot(dout) #(D,N).(N,M) -> (D,M)
    db = dout.sum(axis=0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    step_func = (x>0).astype(int)
    dx = dout * step_func
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################

        sample_mean = x.mean(axis=0)
        sample_variance = x.var(axis=0)
        x_normalized = (x - sample_mean) / np.sqrt(sample_variance + eps)
        out = x_normalized * gamma + beta
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_variance

        cache = {}
        cache['gamma'] = gamma
        cache['beta'] = beta
        cache['sample_mean'] = sample_mean
        cache['sample_variance'] = sample_variance
        cache['x'] = x
        cache['epsilon'] = eps
        cache['x_normalized'] = x_normalized

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        out = x_normalized * gamma + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    gamma = cache['gamma']
    beta = cache['beta']
    sample_mean = cache['sample_mean']
    sample_variance = cache['sample_variance']
    x = cache['x']
    eps = cache['epsilon']
    x_normalized = cache['x_normalized']

    N = x.shape[0]

    x_shifted = x - sample_mean
    sigma = np.sqrt(sample_variance + eps)

    term1 = (dout - dout.sum(axis=0) / N) / sigma
    
    factor = dout.T.dot(x_shifted).diagonal()
    term2 = x_shifted * factor / sigma**3 / N

    dx = (term1 - term2) * gamma

    dgamma = x_normalized.T.dot(dout).diagonal()
    dbeta = dout.sum(axis=0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # probability to keep a neuron
        keep_prob = 1 - p
        # inverse dropout: divide by prob to preserve the expectation value
        mask = np.random.binomial(n=1, p=keep_prob, size=x.shape) / keep_prob
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']

    Hp = int(1 + (H + 2 * pad - HH) / stride)
    Wp = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros(shape=(N, F, Hp, Wp))

    padded_x = np.pad(x, pad_width=((0,), (0,), (pad,), (pad,)), mode='constant')

    # Loop over the input position (with padding)
    # for n in range(N):
    #     for ip in range(Hp):
    #         for jp in range(Wp):
    #             # Loop over the filters
    #             for f in range(F):
    #                 # Dot product for given filter at location
    #                 ip_loc = ip * stride
    #                 jp_loc = jp * stride
    #                 out[n,f,ip,jp] = np.sum(padded_x[n, :, ip_loc:(ip_loc+HH), jp_loc:(jp_loc+WW)] * w[f]) + b[f]

    # Loop over output locations
    for ip in range(Hp):
        for jp in range(Wp):
            # Loop over filters
            for f in range(F):
                ip_loc = ip * stride
                jp_loc = jp * stride
                prod = padded_x[:, :, ip_loc:(ip_loc+HH), jp_loc:(jp_loc+WW)] \
                       * w[f]
                out[:,f,ip,jp] = np.sum(prod, axis=(1,2,3)) + b[f]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']

    # Output height/width
    Hp = int(1 + (H + 2 * pad - HH) / stride)
    Wp = int(1 + (W + 2 * pad - WW) / stride)

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)

    # dout has shape: Output data, of shape (N, F, Hp, Wp)
    db = dout.sum(axis=(0,2,3))

    # (f,ip,jp) are output coordinates
    for f in range(F):
        for ip in range(Hp):
            for jp in range(Wp):
                ip_loc = ip * stride
                jp_loc = jp * stride

                #x_padded_i_range = range(ip_loc, ip_loc+HH)
                #x_padded_j_range = range(jp_loc, jp_loc+WW)

                x_ip_loc = ip_loc - pad
                x_jp_loc = jp_loc - pad

                #x_i_range = range(max(0, x_ip_loc), min(x_ip_loc+HH, H))
                #x_j_range = range(max(0, x_jp_loc), min(x_jp_loc+WW, W))

                # It's important not to remove out-of-x-range coordinates
                # at this point, because that will incorrectly offset the
                # filter coordinates wi,wj used below.
                x_i_range = range(x_ip_loc, x_ip_loc+HH)
                x_j_range = range(x_jp_loc, x_jp_loc+WW)

                for wi, i in enumerate(x_i_range):
                    if i < 0 or i >= H: continue
                    for wj, j in enumerate(x_j_range):
                        if j < 0 or j >= W: continue
                        for c in range(C):
                            dx[:,c,i,j] += dout[:,f,ip,jp] * w[f,c,wi,wj]
                            dw[f,c,wi,wj] += dout[:,f,ip,jp].dot(x[:,c,i,j])

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape

    pH = pool_param['pool_height']
    pW = pool_param['pool_width']
    stride = pool_param['stride']

    outH = int(1 + (H - pH) / stride)
    outW = int(1 + (W - pW) / stride)

    out = np.zeros(shape=(N, C, outH, outW))

    for i in range(outH):
        for j in range(outW):
            xi = i * stride
            xj = j * stride
            window = x[:, :, xi:(xi+pH), xj:(xj+pW)]
            out[:,:,i,j] = np.max(window, axis=(2,3))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pH = pool_param['pool_height']
    pW = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, outH, outW = dout.shape
    
    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(outH):
                for j in range(outW):
                    xi = i * stride
                    xj = j * stride

                    # There's probably a nicer way to do it, without the
                    # n,c loops, but unravel_index is weird.
                    window = x[n,c,xi:(xi+pH),xj:(xj+pW)]
                    max_arg = np.unravel_index(np.argmax(window), (pH,pW))
                    dx[n, c, xi+max_arg[0], xj+max_arg[1]] += dout[n,c,i,j]


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def transpose_channel_axis_to_end(x):
    """
    Given x with shape (N, C, H, W), transpose to (N, H, W, C).
    """
    return x.transpose((0,2,3,1))

def transpose_channel_axis_back(x):
    """
    Given x with shape (N, H, W, C), transpose back to (N, C, H, W).
    """
    return x.transpose((0,3,1,2))

def spatial_batchnorm_normalize(x, gamma, beta, eps, mean, variance):
    """
    Normalize the given data using spatial batchnorm.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - eps: Constant for numeric stability
    - mean: Array of shape (C,) giving feature mean to use for normalization
    - variance: Array of shape (C,) giving feature variance to use
    
    Returns a tuple of:
    - normalized data, shaped (N, C, H, W)
    - normalized data before applying gamma and beta, and shaped (N, H, W, C)
    """
    # Make the ordering (N,H,W,C), so we can act on the channel axis and
    # broadcast over the rest.
    x_t = transpose_channel_axis_to_end(x)
    x_t_normalized = (x_t - mean) / np.sqrt(variance + eps) 
    out_t = x_t_normalized * gamma + beta
    out = transpose_channel_axis_back(out_t)
    return out, x_t_normalized

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    D = C = x.shape[1]
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = x.mean(axis=(0,2,3))
        sample_variance = x.var(axis=(0,2,3))
        out, x_t_normalized = spatial_batchnorm_normalize(
            x, gamma, beta, eps, sample_mean, sample_variance)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_variance

        cache = {}
        cache['gamma'] = gamma
        cache['beta'] = beta
        cache['sample_mean'] = sample_mean
        cache['sample_variance'] = sample_variance
        cache['x'] = x
        cache['epsilon'] = eps
        cache['x_t_normalized'] = x_t_normalized
    elif mode == 'test':
        out = spatial_batchnorm_normalize(x, gamma, beta, eps,
                                          running_mean, running_var)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    gamma = cache['gamma']
    beta = cache['beta']
    sample_mean = cache['sample_mean']
    sample_variance = cache['sample_variance']
    x = cache['x']
    eps = cache['epsilon']
    x_t_normalized = cache['x_t_normalized']

    x_t = transpose_channel_axis_to_end(x)
    dout_t = transpose_channel_axis_to_end(dout)

    # Total number of elements we are computing statistics over
    totalN = x_t.shape[0] * x_t.shape[1] * x_t.shape[2]

    x_shifted_t = x_t - sample_mean
    sigma = np.sqrt(sample_variance + eps)

    term1 = (dout_t - dout_t.sum(axis=(0,1,2)) / totalN) / sigma
    
    # Dot product over the statistics axes (N,H,W)
    factor = np.tensordot(dout_t,
                          x_shifted_t,
                          axes=((0,1,2),(0,1,2))).diagonal()
    term2 = x_shifted_t * factor / sigma**3 / totalN

    dx_t = (term1 - term2) * gamma
    dx = transpose_channel_axis_back(dx_t)

    dgamma = np.tensordot(x_t_normalized, dout_t,
                          axes=((0,1,2),(0,1,2))).diagonal()
    dbeta = dout.sum(axis=(0,2,3))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
