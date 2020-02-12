import numpy as np
import tensorflow as tf

from models.layers import DiscreteLatent
from models.tfmodel import TFModel
from helpers import tf_helpers, paramspec


class DCN(TFModel):
    """
    An abstract class for deriving image compression models.

    # Attributes set-up by the abstract class:
      x                    - model input
      patch_size           - patch size
      latent_bpf           - number of bits per feature of the latent representation
      train_codebook       - whether the codebook
      codebook             - the quantization code book (TF)
      entropy_weight       - entropy regularization strength for model loss
      default_val_is_train - used to set default value for the 'is_training' flag during model inference
                             (useful for models with batch normalization)
      scale_latent         - bool flag indicating scaling of the latent representation
      use_batchnorm        - bool flag indicating the use of batch norm in the model

      weights              - soft quantization weights (TF)
      histogram            - latent space histogram based on soft quantization (TF)
      entropy              - entropy estimation (TF)

    # Attributes that need to be set-up by the derived classes:
      y
      latent_pre           - latent representation before quantization
      latent_post          - latent representation after quantization
      latent_shape         - shape of the latent tensor (before flattening)
      n_latent             - dimensionality of the latent representation
      _h                   - hyper parameters

    For setting up quantization, use the provided self._setup_latent_space method - it will create the latent_pre and
    latent_post attributes.
    """

    def __init__(self, label=None, patch_size=128, latent_bpf=5, rounding='soft-codebook', train_codebook=False, entropy_weight=250, scale_latent=True, use_batchnorm=False, loss_metric='L2', **kwargs):
        """
        Creates a forensic analysis network.

        :param sess: TF session or None (creates a new one)
        :param graph: TF graph or None (creates a new one)
        :param label: a suffix for the name scope of the model
        """
        super().__init__(label)

        # Parameter sanitization
        self._h = paramspec.ParamSpec({
            'latent_bpf': (5, int, (1, 8)),
            'train_codebook': (False, bool, None),
            'entropy_weight': (250, float, (0, 1e6)),
            'scale_latent': (True, bool, None),
            'use_batchnorm': (False, bool, None),
            'loss_metric': ('L2', str, {'L2'}),
            'rounding': ('soft', str, {'identity', 'soft', 'soft-codebook', 'sin'})
        })
        params = locals()
        self._h.update(**{k: params[k] for k in self._h.keys()})
        self.patch_size = patch_size

        self.x = tf.keras.Input(dtype=tf.float32, shape=(patch_size, patch_size, 3))

        # Prepare the quantization layer        
        self.discrete_latent = DiscreteLatent(self._h.rounding, self._h.latent_bpf)

        # Construct the actual model -------------------------------------------------------------------------------
        self.construct_model(**kwargs)
        self._has_attributes(['y', '_model', '_encoder', '_decoder'])
        
        # Add entropy estimation and model optimization operations -------------------------------------------------
        with tf.name_scope('{}/optimization'.format(self.scoped_name)):

            # Loss and SSIM
            self.ssim = lambda a, b: tf.reduce_mean(tf.image.ssim(a, b, max_val=1))
            
            if loss_metric == 'L2':
                def mse_entropy(image_target, image_compressed, entropy):
                    return tf.nn.l2_loss(image_target - image_compressed) + self._h.entropy_weight * entropy
                self.loss = mse_entropy
            else:
                raise NotImplementedError('Loss metric {} not supported yet.'.format(loss_metric))
                        
            # Optimization
            self.optimizer = tf.keras.optimizers.Adam()

    def construct_model(self, params):
        raise NotImplementedError('Not implemented!')

    def reset_performance_stats(self):
        self.performance = {
            'loss': {'training': [], 'validation': []},
            'entropy': {'training': [], 'validation': []},
            'ssim': {'training': [], 'validation': []},
            'psnr': {'training': [], 'validation': []}
        }

    # def get_tf_histogram(self, batch_x, is_training=None):
    #     with self.graph.as_default():
    #         feed_dict = {
    #             self.x if not self.use_nip_input else self.nip_input: batch_x,
    #         }

    #         if hasattr(self, 'is_training'):
    #             feed_dict[self.is_training] = is_training if is_training is not None else self.default_val_is_train

    #         return self.sess.run(self.histogram, feed_dict=feed_dict)

    def compress(self, batch_x):
        """
        Compress an input batch to a quantized latent representation.

        :param batch_x: Input tensor (N, H, W, 3:rgb) or (N, H, W, 4:rggb) for RAW data chained through a NIP
        :param is_training: can be used to override the default 'is_training' flag (may be useful for models with BN)
        :param direct: controls whether the input is a RAW image (chained through a NIP) or direct RGB input
        :return:
        """
        return self._encoder(np.expand_dims(batch_x, axis=0) if batch_x.ndim == 3 else batch_x)[0]
        
    def decompress(self, batch_z):
        """
        Decompress a batch of images from their quantized latent representations.
        :param batch_z: batch of quantized latent values
        :param is_training: can be used to override the default 'is_training' flag (may be useful for models with BN)
        :return:
        """
        return self._decoder(np.expand_dims(batch_z, axis=0) if batch_z.ndim == 3 else batch_z)
            
    def process(self, batch_x):
        """
        Process the image through the whole model (encoder-quantization-decoder).
        :param batch_x: Input tensor (N, H, W, 3:rgb) or (N, H, W, 4:rggb) for RAW data chained through a NIP
        :param dropout_keep_prob: set keep probability in case of using Dropout
        :param is_training: can be used to override the default 'is_training' flag (may be useful for models with BN)
        :param direct: controls whether the input is a RAW image (chained through a NIP) or direct RGB input
        """
        return self._model(batch_x)[0]

    def training_step(self, batch_x, learning_rate=1e-4):
        """
        Make a single training step and return current loss. Only the FAN model is updated.
        """
        with tf.GradientTape() as tape:
            batch_Y, entropy = self._model(batch_x)
            loss = self.loss(batch_x, batch_Y, entropy)
            ssim = self.ssim(tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_Y))

        self.optimizer.lr.assign(learning_rate)
        grads = tape.gradient(loss, self._model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
        return {
                'loss': np.sqrt(2 * loss),  # The L2 loss in TF is computed differently (half of non-square rooted norm)
                'ssim': ssim,
                'entropy': entropy
            }

    def compression_stats(self, patch_size=None, n_latent_bytes=None):
        """
        Get expected compression stats for the model:
            - data rate
            - bits per pixel (bpp)
            - bits per feature (bpf)
            - bytes

        :param patch_size: Can be used to override the default input size
        :param n_latent_bytes: Can be used to override the default bpf; Specified per feature.
        :return:
        """

        n_latent_bytes = n_latent_bytes or self._h.latent_bpf / 8

        ps = patch_size or self.patch_size        
        if ps is None:
            raise ValueError('Patch size not specified!')
            
        bitmap_size = ps * ps * 3
        return {
            'rate': bitmap_size / (n_latent_bytes * self.n_latent),
            'bpp': 8 * self.n_latent * n_latent_bytes / (ps * ps),
            'bpf': 8 * n_latent_bytes,
            'bytes': self.n_latent * n_latent_bytes
        }
    
    def summary(self):
        return 'DCN with a {}-dim {}-bpf latent representation [{:,} params]'.format(
            'x'.join(str(x) for x in self.latent_shape), 
            self._h.latent_bpf,
            self.count_parameters()
        )
    
    @property
    def model_code(self):
        if not hasattr(self, 'n_latent'):
            raise ValueError('The model does not report the latent space dimensionality.')
        
        return '{}-{}C'.format(type(self).__name__, self._h.n_features)        

    def get_hyperparameters(self):
        return self._h.to_json()

    def get_codebook(self):
        return self.discrete_latent.quantization.codebook.numpy().reshape((-1,))


class TwitterDCN(DCN):
    """
    Auto-encoder architecture described in:
    [1] L. Theis, W. Shi, A. Cunningham, and F. Huszár, “Lossy Image Compression with Compressive
    coders,” Mar. 2017.
    """

    def construct_model(self, n_features=32, activation='leaky_relu'):

        # Define expected hyper parameters and their values ------------------------------------------------------------
        self._h.add({
            'n_features': (96, int, (4, 128)),
            'activation': ('leaky_relu', str, set(tf_helpers.activation_mapping.keys()))
        })

        params = locals()
        self._h.update(**{k: params[k] for k in self._h.keys() if k in params})

        if self.patch_size is None:
            self.latent_shape = (None, None, self._h.n_features)
            self.n_latent = None
        else:
            self.latent_shape = (self.patch_size // 8, self.patch_size // 8, self._h.n_features)
            self.n_latent = int(np.prod(self.latent_shape))

        activation = tf_helpers.activation_mapping[self._h.activation]

        # Encoder ------------------------------------------------------------------------------------------------------

        net = 2 * (self.x - 0.5)

        net = tf.keras.layers.Conv2D(64, 5, 2, padding='SAME', activation=activation)(net)
        net = tf.keras.layers.Conv2D(128, 5, 2, padding='SAME', activation=None)(net)

        net_relu = tf.nn.leaky_relu(net)
        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=activation)(net_relu)
        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=None)(resnet)
        net = tf.add(net, resnet)

        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=activation)(net)
        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=None)(resnet)
        net = tf.add(net, resnet)

        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=activation)(net)
        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=None)(resnet)
        net = tf.add(net, resnet)

        net = tf.keras.layers.Conv2D(self._h.n_features, 5, 2, padding='SAME', activation=None)(net)

        # Latent space -------------------------------------------------------------------------------------------------

        self.latent, self.entropy = self.discrete_latent(net)

        # Decoder ------------------------------------------------------------------------------------------------------

        self.latent_input = tf.keras.Input(dtype=tf.float32, shape=self.latent.shape[1:])

        inet = tf.keras.layers.Conv2D(512, 3, 1, padding='SAME', activation=None)(self.latent_input)
        inet = tf.nn.depth_to_space(inet, 2)

        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=activation)(inet)
        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=None)(resnet)
        inet = tf.add(inet, resnet)

        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=activation)(inet)
        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=None)(resnet)
        inet = tf.add(inet, resnet)

        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=activation)(inet)
        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=None)(resnet)
        inet = tf.add(inet, resnet)

        inet = tf.keras.layers.Conv2D(256, 3, 1, padding='SAME', activation=activation)(inet)
        inet = tf.nn.depth_to_space(inet, 2)

        inet = tf.keras.layers.Conv2D(12, 3, 1, padding='SAME', activation=None)(inet)
        inet = tf.nn.depth_to_space(inet, 2)

        y = (inet + 1) / 2

        # Overwrite the output to guarantee correct data range and maintain gradient propagation
        self.y = tf.stop_gradient(tf.clip_by_value(y, 0, 1) - y) + y
        
        # Create separate models to enable separate encoding / decoding steps
        self._encoder = tf.keras.Model(inputs=self.x, outputs=[self.latent, self.entropy], name='encoder')
        self._decoder = tf.keras.Model(inputs=self.latent_input, outputs=self.y, name='decoder')

        # Combine the models to enable compression simulation, training and 1-step model saving / loading
        latent, entropy = self._encoder(self.x)
        self._model = tf.keras.Model(inputs=self.x, outputs=[self._decoder(latent), entropy], name='codec')

    @property
    def model_code(self):
        parameter_summary = []

        parameter_summary.append(self._h.rounding)
        parameter_summary.append(
            'Q+{}bpf'.format(self._h.latent_bpf) if self._h.train_codebook else 'Q-{}bpf'.format(self._h.latent_bpf))
        parameter_summary.append('S+' if self._h.scale_latent else 'S-')
        if self._h.entropy_weight is not None:
            parameter_summary.append('H+{:.2f}'.format(self._h.entropy_weight))

        return '{}/{}'.format(super().model_code, '_'.join(parameter_summary))
