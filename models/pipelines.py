import sys
import inspect
import numpy as np
import tensorflow as tf

from collections import OrderedDict

from models.tfmodel import TFModel
from helpers import tf_helpers, paramspec
from helpers.utils import upsampling_kernel, bilin_kernel, gamma_kernels


@tf.function
def mse(a, b):
    return tf.reduce_mean(tf.math.pow(255 * a - 255 * b, 2.0))

@tf.function
def mae(a, b):
    return tf.reduce_mean(tf.math.abs(255 * a - 255 * b))

@tf.function
def ssiml(a, b):
    return 255 * (1 - tf.image.ssim(a, b, 1.0))

@tf.function
def msssiml(a, b):
    return 255 * (1 - tf.image.ssim_multiscale(a, b, 1.0))


class NIPModel(TFModel):
    """
    Abstract class for implementing neural imaging pipelines. Specific classes are expected to implement the
    'construct_model' method that builds the model, and 'parameters' method which lists its parameters. See existing
    classes for examples.
    """

    def __init__(self, loss_metric='L2', patch_size=None, label=None, reuse_placeholders=None, in_channels=4, out_shape_mx=2, **kwargs):
        """
        Base constructor with common setup.

        :param sess: TF session or None (creates a new one)
        :param graph: TF graph or None (creates a new one)
        :param loss_metric: loss metric for NIP optimization (L2, L1, SSIM)
        :param patch_size: Optionally patch size can be given to fix placeholder dimensions (can be None)
        :param label: A string prefix for the model (useful when multiple NIPs are used in a single TF graph)
        :param reuse_placeholders: Give a dictionary with 'x' and 'y' keys if multiple NIPs should use the same inputs
        :param kwargs: Additional arguments for specific NIP implementations
        """
        super().__init__(label)

        # Initialize input placeholders and run 'construct_model' to build the model and
        # setup its output as self.y
        self.y = None  # This will be set up later by child classes

        if reuse_placeholders is not None:
            self.x = reuse_placeholders['x']
            self.y_gt = reuse_placeholders['y']
        else:
            out_patch_size = out_shape_mx * patch_size if patch_size is not None else None
            self.x = tf.keras.Input(dtype=tf.float32, shape=(patch_size, patch_size, in_channels), name='x')
            self.y_gt = tf.keras.Input(dtype=tf.float32, shape=(out_patch_size, out_patch_size, 3), name='y')
        
        self.in_channels = in_channels
        self.out_shape_mx = out_shape_mx
        self.construct_model(**kwargs)

        # Configure loss and model optimization
        self.loss_metric = loss_metric
        self.construct_loss(loss_metric)

    def construct_loss(self, loss_metric):
        y = self.yy if hasattr(self, 'yy') else self.y
        
        # The loss
        if loss_metric == 'L2':
            # self.loss = tf.keras.losses.MeanSquaredError()
            self.loss = mse 
        elif loss_metric == 'L1':
            self.loss = mae
        elif loss_metric == 'SSIM':
            self.loss = ssiml
        elif loss_metric == 'MS-SSIM':
            self.loss = msssiml
        else:
            raise ValueError('Unsupported loss metric!')

        self.optimizer = tf.keras.optimizers.Adam()
    
    def construct_model(self):
        """
        Constructs the NIP model. The method should use self.x as RAW image input, and set self.y as the model output.
        The output is expected to be clipped to [0,1]. For better optimization stability, the model can set self.yy to
        non-clipped output (will be used for gradient computation).

        A string prefix (self.scoped_name) should be used for variables / named scopes to facilitate using multiple NIPs
        in a single TF graph.
        """
        raise NotImplementedError()

    def training_step(self, batch_x, batch_y, learning_rate):
        """
        Make a single training step and return the loss.
        """
        with tf.GradientTape() as tape:

            batch_Y = self._model(batch_x)
            loss = self.loss(batch_Y, batch_y)

        self.optimizer.lr.assign(learning_rate)
        grads = tape.gradient(loss, self._model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
        return loss.numpy()
        
    def process(self, batch_x, is_training=False):
        """
        Develop RAW input and return RGB image.
        """
        if batch_x.ndim == 3:
            batch_x = np.expand_dims(batch_x, 0)
        
        return self._model(batch_x)
    
    def reset_performance_stats(self):
        self.performance = {
            'loss': {'training': [], 'validation': []},
            'psnr': {'validation': []},
            'ssim': {'validation': []},
            'dmse': {'validation': []}
        }

    def get_hyperparameters(self):
        p = {
            'in_channels': self.in_channels,
            'out_shape_mx': self.out_shape_mx 
        }
        if hasattr(self, '_h'):
            p.update(self._h.to_json())
        return p

    @property
    def _input_description(self):
        if self.patch_size_raw is None:
            return '(rgb)' if self.x.shape[-1] == 3 else '(raw)'
        else:
            return 'x'.join(str(x) for x in self.x.shape[1:])

    @property
    def _output_description(self):
        if self.patch_size_rgb is None:
            return '(rgb)' if self.y.shape[-1] == 3 else '(?)'
        else:
            return 'x'.join(str(x) for x in self.y.shape[1:])

    @property
    def patch_size_raw(self):
        return self.x.shape[1:]

    @property
    def patch_size_rgb(self):
        return self.y.shape[1:]

    def summary(self):
        return '{:s} : {} -> {}'.format(super().summary(), self._input_description, self._output_description)

class UNet(NIPModel):
    """
    The UNet model, rewritten from scratch for TF 2.x
    Originally adapted from https://github.com/cchen156/Learning-to-See-in-the-Dark
    """
        
    def construct_model(self, **kwargs):
        # Define expected hyper parameters and their values ------------------------------------------------------------
        self._h = paramspec.ParamSpec({
            'n_steps': (5, int, (2, 6)),
            'activation': ('leaky_relu', str, set(tf_helpers.activation_mapping.keys()))
        })

        self._h.update(**kwargs)
        lrelu = tf_helpers.activation_mapping[self._h.activation]
        
        # lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

        _layers = OrderedDict()
        _tensors = OrderedDict()
        _tensors['ep0'] = self.x

        # Construct the encoder
        for n in range(1, self._h.n_steps + 1):
            _layers['ec{}1'.format(n)] = tf.keras.layers.Conv2D(32 * 2**(n-1), [3, 3], activation=lrelu, padding='SAME')
            _layers['ec{}2'.format(n)] = tf.keras.layers.Conv2D(32 * 2**(n-1), [3, 3], activation=lrelu, padding='SAME')
            _tensors['ec{}1'.format(n)] = _layers['ec{}1'.format(n)](_tensors['ep{}'.format(n-1)])
            _tensors['ec{}2'.format(n)] = _layers['ec{}2'.format(n)](_tensors['ec{}1'.format(n)])

            if n < self._h.n_steps:
                _layers['ep{}'.format(n)] = tf.keras.layers.MaxPool2D([2, 2], padding='SAME')
                _tensors['ep{}'.format(n)]  = _layers['ep{}'.format(n)](_tensors['ec{}2'.format(n)])
            
        # Easy access to encoder output via a recursive relation
        _tensors['dc02'] = _tensors['ec{}2'.format(self._h.n_steps)]

        # Construct the decoder
        for n in range(1, self._h.n_steps):
            _layers['dct{}'.format(n)] = tf.keras.layers.Conv2DTranspose(32 * 2**(self._h.n_steps - n - 1), [2, 2], [2, 2], padding='SAME')
            _layers['dcat{}'.format(n)] = tf.keras.layers.Concatenate()
            _layers['dc{}1'.format(n)] = tf.keras.layers.Conv2D(32 * 2**(self._h.n_steps - n - 1), [3, 3], activation=lrelu, padding='SAME')
            _layers['dc{}2'.format(n)] = tf.keras.layers.Conv2D(32 * 2**(self._h.n_steps - n - 1), [3, 3], activation=lrelu, padding='SAME')

            _tensors['dct{}'.format(n)] = _layers['dct{}'.format(n)](_tensors['dc{}2'.format(n-1)])
            _tensors['dcat{}'.format(n)] = _layers['dcat{}'.format(n)]([_tensors['dct{}'.format(n)], _tensors['ec{}2'.format(self._h.n_steps - n)]])
            _tensors['dc{}1'.format(n)] = _layers['dc{}1'.format(n)](_tensors['dcat{}'.format(n)])
            _tensors['dc{}2'.format(n)] = _layers['dc{}2'.format(n)](_tensors['dc{}1'.format(n)])

        # Final step to render the RGB image
        _layers['dc{}'.format(self._h.n_steps)] = tf.keras.layers.Conv2D(12, [3, 3], padding='SAME')
        _tensors['dc{}'.format(self._h.n_steps)] =_layers['dc{}'.format(self._h.n_steps)](_tensors['dc{}2'.format(self._h.n_steps - 1)])
        _tensors['dts'] = tf.nn.depth_to_space(_tensors['dc{}'.format(self._h.n_steps)], 2)

        # Add NIP outputs
        self.yy = _tensors['dts']
        self.y = tf.clip_by_value(_tensors['dts'], 0, 1)

        # Construct the Keras model
        self._model = tf.keras.Model(inputs=[self.x], outputs=[self.y], name='unet')

class INet(NIPModel):
    """
    A neural pipeline which replicates the steps of a standard imaging pipeline.
    """
    
    def construct_model(self, random_init=False, kernel=5, trainable_upsampling=False, cfa_pattern='gbrg'):
        self.trainable_upsampling = trainable_upsampling
        self.cfa_pattern = cfa_pattern

        # Initialize the upsampling kernel
        upk = upsampling_kernel(cfa_pattern)

        if random_init:
            # upk = np.random.normal(0, 0.1, (4, 12))
            dmf = np.random.normal(0, 0.1, (kernel, kernel, 3, 3))
            gamma_d1k = np.random.normal(0, 0.1, (3, 12))
            gamma_d1b = np.zeros((12, ))
            gamma_d2k = np.random.normal(0, 0.1, (12, 3))
            gamma_d2b = np.zeros((3,))
            srgbk = np.eye(3)
        else:    
            # Prepare demosaicing kernels (bilinear)
            dmf = bilin_kernel(kernel)

            # Prepare gamma correction kernels (obtained from a pre-trained toy model)
            gamma_d1k, gamma_d1b, gamma_d2k, gamma_d2b = gamma_kernels()

            # Example sRGB conversion table
            srgbk = np.array([[ 1.82691061, -0.65497452, -0.17193617],
                                [-0.00683982,  1.33216381, -0.32532394],
                                [ 0.06269717, -0.40055895,  1.33786178]]).transpose()

        # Up-sample the input back the full resolution
        h12 = tf.keras.layers.Conv2D(12, 1, kernel_initializer=tf.constant_initializer(upk), use_bias=False, activation=None, trainable=trainable_upsampling)(self.x)

        # Demosaicing
        pad = (kernel - 1) // 2
        bayer = tf.nn.depth_to_space(h12, 2)
        bayer = tf.pad(bayer, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), 'REFLECT')
        rgb = tf.keras.layers.Conv2D(3, kernel, kernel_initializer=tf.constant_initializer(dmf), use_bias=False, activation=None, padding='VALID')(bayer)

        # Color space conversion
        srgb = tf.keras.layers.Conv2D(3, 1, kernel_initializer=tf.constant_initializer(srgbk), use_bias=False, activation=None)(rgb,)

        # Gamma correction
        rgb_g0 = tf.keras.layers.Conv2D(12, 1, kernel_initializer=tf.constant_initializer(gamma_d1k), bias_initializer=tf.constant_initializer(gamma_d1b), use_bias=True, activation=tf.keras.activations.tanh)(srgb)
        self.yy = tf.keras.layers.Conv2D(3, 1, kernel_initializer=tf.constant_initializer(gamma_d2k), bias_initializer=tf.constant_initializer(gamma_d2b), use_bias=True, activation=None)(rgb_g0)
    
        self.y = tf.clip_by_value(self.yy, 0, 1, name='{}/y'.format(self.scoped_name))
        self._model = tf.keras.Model(inputs=[self.x], outputs=[self.y])


class DNet(NIPModel):
    """
    Neural imaging pipeline adapted from a joint demosaicing-&-denoising model:
    Gharbi, Michaël, et al. "Deep joint demosaicking and denoising." ACM Transactions on Graphics (TOG) 35.6 (2016): 191.
    """

    def construct_model(self, n_layers=15, kernel=3, n_features=64):

        k_initializer = tf.keras.initializers.VarianceScaling

        # Initialize the upsampling kernel
        upk = upsampling_kernel()

        # Padding size
        pad = (kernel - 1) // 2

        # Convolutions on the sub-sampled input tensor
        deep_x = self.x
        for r in range(n_layers):
            deep_y = tf.keras.layers.Conv2D(12 if r == n_layers - 1 else n_features, kernel, activation=tf.keras.activations.relu, padding='VALID', kernel_initializer=k_initializer)(deep_x)
            deep_x = tf.pad(deep_y, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), 'REFLECT')

        # Up-sample the input
        h12 = tf.keras.layers.Conv2D(12, 1, kernel_initializer=tf.constant_initializer(upk), use_bias=False, activation=None, trainable=False)(self.x)
        bayer = tf.nn.depth_to_space(h12, 2)

        # Upscale the conv. features and concatenate with the input RGB channels
        features = tf.nn.depth_to_space(deep_x, 2)
        bayer_features = tf.concat((features, bayer), axis=3)            

        # Project the concatenated 6-D features (R G B bayer from input + 3 channels from convolutions)
        pu = tf.keras.layers.Conv2D(n_features, kernel, kernel_initializer=k_initializer, use_bias=True, activation=tf.keras.activations.relu, padding='VALID', bias_initializer=tf.zeros_initializer)(bayer_features)

        # Final 1x1 conv to project each 64-D feature vector into the RGB colorspace
        pu = tf.pad(pu, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), 'REFLECT')

        self.yy = tf.keras.layers.Conv2D(3, 1, kernel_initializer=tf.ones_initializer, use_bias=False, activation=None, padding='VALID')(pu)
        self.y = tf.clip_by_value(self.yy, 0, 1, name='{}/y'.format(self.scoped_name))
        self._model = tf.keras.Model(inputs=[self.x], outputs=[self.y])


supported_models = [name for name, obj in inspect.getmembers(sys.modules[__name__]) if type(obj) is type and issubclass(obj, NIPModel) and name != 'NIPModel']


class ONet(NIPModel):
    """
    Dummy pipeline for RGB manipulation training.
    """

    def construct_model(self):
        self.x = self.y_gt
        self.yy = self.y_gt
        self.y = self.y_gt
