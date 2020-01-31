import numpy as np
import tensorflow as tf

import helpers.tf_helpers
from models.tfmodel import TFModel
from models.layers import ConstrainedConv2D
from helpers import utils, paramspec, tf_helpers


class FAN(TFModel):
    """
    A forensic analysis network with the following architecture:

    1. A constrained conv layer (learned residual filter)
    2. N x standard conv layers
    3. A 1x1 conv layer
    4. GAP for feature extraction
    5. 2 hidden fully connected layers
    6. Output layer with K classes
    """

    def __init__(self, n_classes, patch_size=None, label=None, n_filters=32, n_fscale=2, n_convolutions=4, kernel=5, dropout=0.0, use_gap=True, activation='leaky_relu'):
        """
        Creates a forensic analysis network.

        :param n_classes: the number of output classes
        :param x: input tensor
        :param nip_input: input to the NIP (if part of a larger model) or None
        :param n_filters: number of output features for the first conv layer
        :param n_fscale: multiplier for the number of output features in successive conv layers
        :param n_convolutions: the number of standard conv layers
        :param kernel: conv kernel size
        :param dropout: dropout rate for fully connected layers
        :param use_gap: whether to use a GAP or to reshape the final conv tensor
        """
        super().__init__(label)
        self.n_classes = n_classes

        # Set-up and validate hyper-parameters            
        self._h = paramspec.ParamSpec({
            'n_filters': (32, int, (4, 128)),
            'n_fscale': (2, float, (0.25, 4)),
            'n_convolutions': (4, int, (1, 32)),
            'kernel': (5, int, (3, 11)),
            'dropout': (0, float, (0, 1)),
            'use_gap': (False, bool, None),
            'activation': ('leaky_relu', str, set(tf_helpers.activation_mapping.keys()))
        })
        params = locals()
        self._h.update(**{k: params[k] for k in self._h.keys()})

        self.x = tf.keras.Input(dtype=tf.float32, shape=(patch_size, patch_size, 3))
            
        # Setup a GT placeholder
        self.y_gt = tf.keras.Input(dtype=tf.int32, shape=(None,))
        
        # Basic parameters
        activation = tf_helpers.activation_mapping[self._h.activation]

        # Constrained convolution with a learned residual filter
        net = ConstrainedConv2D()(self.x)

        # Standard convolutional layers
        for conv_id in range(self._h.n_convolutions):
            net = tf.keras.layers.Conv2D(n_filters, [self._h.kernel, self._h.kernel], activation=activation)(net)
            net = tf.keras.layers.MaxPool2D([2, 2])(net)
            n_filters = int(n_filters * self._h.n_fscale)

        # Final 1 x 1 convolution
        net = tf.keras.layers.Conv2D(n_filters // n_fscale, [1, 1], activation=activation)(net)

        # GAP / Feature formation
        if use_gap:
            net = tf.keras.layers.GlobalAveragePooling2D()(net)
        else:
            net = tf.keras.layers.Flatten()(net)

        # Remember extracted features for future debugging
        self.features = net

        # Fully-connected classifier
        net = tf.keras.layers.Dense(512, activation=activation)(net)
        if dropout > 0: net = tf.keras.layers.Dropout(dropout)(net)
        
        net = tf.keras.layers.Dense(128, activation=activation)(net)        
        if dropout > 0: net = tf.keras.layers.Dropout(dropout)(net)
        
        self.y = tf.keras.layers.Dense(n_classes, activation=tf.keras.activations.softmax)(net)

        self._model = tf.keras.Model(inputs=self.x, outputs=self.y)        
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def reset_performance_stats(self):
        self.performance = {
            'loss': {'training': [], 'validation': []},
            'accuracy': {'validation': []},
            'confusion': [],
        }

    def process(self, batch_x, direct=True):
        """
        Returns the predicted class for an image batch. The input is fed to the NIP if the model is chained properly.
        """
        return self._model(batch_x)

    def process_and_decide(self, batch_x, with_confidence=False):
        """
        Returns class probabilities for an image batch. The input is fed to the NIP if the model is chained properly.
        """
        probs = self._model(batch_x)
        if with_confidence:
            return probs.numpy().argmax(axis=1), probs.numpy().max(axis=1)
        else:
            return probs.numpy().argmax(axis=1)
    
    def process_with_loss(self, batch_x, batch_y):
        """
        Returns the predicted class and loss for an image batch.
        """
        probabilities = self._model(batch_x)
        return probabilities, self.loss(batch_y, probabilities)
    
    def training_step(self, batch_x, batch_y, learning_rate):
        """
        Make a single training step and return current loss. Only the FAN model is updated.
        """
        with tf.GradientTape() as tape:
            batch_Y = self._model(batch_x)
            loss = self.loss(batch_y, batch_Y)

        self.optimizer.lr.assign(learning_rate)
        grads = tape.gradient(loss, self._model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
        return loss

    def __repr__(self):
        extra_params = ','.join('{}={}'.format(k, '"{}"'.format(v) if isinstance(v, str) else v) for k, v in self._h.changed_params().items())
        if len(extra_params) > 0:
            extra_params = ','+extra_params
        return '{}(n_classes={}{})'.format(self.class_name, self.n_classes, extra_params)

    def summary(self):
        return '{kernel}x{kernel} CNN: 1+{conv}+1 conv layers {gap}+ 2 fc layers [{params:,} parameters]'.format(
            kernel=self._h.kernel, 
            conv=self._h.n_convolutions, 
            gap='+ (GAP) ' if self._h.use_gap else '',
            params=self.count_parameters())
