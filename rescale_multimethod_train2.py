import numpy as np
import tensorflow as tf

from helpers import dataset, utils, plots, stats
from helpers import tf_helpers as tfh
from bayesian.sfp import SFP

# tfh.disable_warnings()
# tfh.disable_gpu()
utils.setup_logging()

data = dataset.Dataset('native12k', load='y', n_images=1000, v_images=1000)

scales = (1.0, 1.75) # (0.8, 1.4), (1.5, 2.0)
epochs = 10000
batch_multip = 4
batch_size = 10
patch_size = 128
n_classes = 31
methods = ['nearest', 'bilinear', 'bicubic', 'lanczos3']

classes = np.linspace(*scales, num=n_classes)

print(f'{n_classes}: {classes.tolist()}')

n_batches = data.count_training // batch_size

# Model
model = SFP(c_filters=(32, 32, 32, 32),
            d_filters=(32, 16, n_classes), kernel=5,
            activation='leaky_relu', trainable_residual=True,
            drop=0.1, append_rgb=False)
opt = tf.keras.optimizers.Adam(1e-3)
loss_op = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

performance = {'loss': {'training': []}}

with utils.progress_bar(epochs, 'Traing') as pbar:
    for epoch in range(epochs):

        losses = 0

        for batch_id in range(n_batches):
            batch_y = data.next_training_batch(batch_id, batch_size,
                                               patch_size)
            sf = tf.random.uniform((1,), *scales)
            method = np.random.choice(methods)

            batch_yy = tf.image.resize(batch_y,
                                       [int(sf * patch_size),
                                       int(sf * patch_size)],
                                       method=method)

            class_id = stats.quantize(sf.numpy(), classes, True)
            batch_sf = np.repeat(class_id, batch_size).reshape((-1, 1))

            with tf.GradientTape() as tape:
                loss = loss_op(batch_sf, model(batch_yy, training=True))

            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

            # Update loss counter
            losses += loss.numpy()

        performance['loss']['training'].append(losses / n_batches)

        pbar.set_postfix(loss=losses / n_batches)
        pbar.update(1)

model.save_weights('model_weights/bayesian_scaleFactor_multAlgo2.h5')

