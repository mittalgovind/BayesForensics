# -*- coding: utf-8 -*-
"""
Provides a Dataset class that loads full resolution training images and samples from them randomly. (See class docs.)
"""
import os
import numpy as np
from helpers import loading
from helpers.loading import sample_patch


class Dataset(object):
    def __init__(
            self,
            data_directory=None,
            *,
            randomize=2468,
            load="xy",
            n_images=120,
            v_images=30,
            c_images=50,
            val_rgb_patch_size=128,
            val_n_patches=20,
            val_discard="flat-aggressive",
            presample_epochs=0,
            train_rgb_patch_size=0,
            data_train=None,
            data_val=None,
    ):
        """
        Represents a [RAW-]RGB dataset for training imaging pipelines. The class preloads full resolution images and
        samples from them when requesting training batches. (Validation images are sampled upon creation.) Patch
        selection takes care of proper alignment between RAW images (represented as half-size 4-channel RGGB stacks) and
        their corresponding rendered RGB versions. Selection can be controlled to prefer certain types of patches (not
        strictly enforced). The following DISCARD modes are available:

        - flat - attempts to discard flat patches based on patch variance (not strict)
        - flat-aggressive - a more aggressive version that avoids patches with variance < 0.01
        - dark-n-textured - avoid dark (mean < 0.35) and textured patches (variance > 0.005)

        Usage examples:
        ---------------

        # Load a RAW -> RGB dataset
        data = Dataset('data/raw/training_data/D90')
        batch_raw, batch_rgb = data.next_training_batch(0, 10, 128, 'flat-aggressive')

        # Load RGB only dataset
        data = Dataset('data/rgb/native12k/', load='y')
        batch_rgb = data.next_training_batch(0, 10, 128, 'flat-aggressive')

        :param data_directory: directory path with RAW-RGB pairs (*.npy & *.png) or only RGB images (*.png)
        :param randomize: randomization seed
        :param load: what data to load: 'xy' load RAW+RGB, 'x' load RAW only, 'y' load RGB only
        :param n_images: number of training images (full resolution)
        :param v_images: number of validation images (patches sampled upon creation)
        :param c_images: number of calibration images (patches sampled upon creation)
        :param val_rgb_patch_size: validation patch size
        :param val_n_patches: number of validation patches to load per full-resolution image
        :param val_discard: patch discard mode (for validation data)
        """
        if not any(load == allowed for allowed in {"xy", "x", "y"}):
            raise ValueError("Invalid X/Y data requested!")

        if train_rgb_patch_size == 0:
            train_rgb_patch_size = val_rgb_patch_size
        elif presample_epochs == 0:
            raise ValueError(
                "Setting training patch size here only works when preloading the patches "
                "(when presample_epochs>0). Otherwise, full resolution images are loaded."
            )

        self.data = {}
        self.files = {}
        self._loaded_data = load
        self._data_directory = data_directory
        self._counts = (
            n_images if presample_epochs == 0 else n_images * presample_epochs,
            v_images,
            c_images,
            val_n_patches,
        )
        self._val_discard = "flat-aggressive"

        self.data["training"] = {}
        self.data["validation"] = {}
        self.data["training"]['y'] = data_train
        self.data["validation"]['y'] = data_val
        self.presample_epochs = 1
        self.patch_size = val_rgb_patch_size

    def __getitem__(self, key):
        if key in ["training", "validation", "calibration"]:
            return self.data[key]
        else:
            raise KeyError("Key: {} not found!".format(key))

    def next_training_batch(
            self, batch_id, batch_size, rgb_patch_size=0, discard="flat",
            max_attempts=25
    ):
        """
        Sample a new batch of training patches.
        :param batch_id: integer from 0 to (#training images // batch_size - 1)
        :param batch_size: integer, self explanatory
        :param rgb_patch_size: patch size (in full-resolution RGB coordinates; RAW patches [RGGB] have half the size)
        :param discard: patch discard mode (for validation data)
        :param max_attempts: maximum number of sampling attempts (if unsuccessful)
        :return: tuple of np arrays (RAW, RGB) or np array (RGB)
        """
        if rgb_patch_size == 0:
            rgb_patch_size = min(128, *self.train_image_shape_rgb[:2])

        if discard is not None and "y" not in self.data["training"]:
            raise ValueError(
                "Cannot discard patches if RGB data is not loaded.")

        if (batch_id + 1) * batch_size > self.count_training:
            raise ValueError(
                "Not enough images for the requested batch_id & batch_size"
            )

        raw_patch_size = rgb_patch_size // 2

        # Allocate memory for the batch
        has_raw = "x" in self._loaded_data
        has_rgb = "y" in self._loaded_data
        bx = (
            np.zeros((batch_size, raw_patch_size, raw_patch_size, 4),
                     dtype=np.float32)
            if has_raw
            else None
        )
        by = (
            np.zeros((batch_size, rgb_patch_size, rgb_patch_size, 3),
                     dtype=np.float32)
            if has_rgb
            else None
        )

        if rgb_patch_size > min(self.train_image_shape_rgb[:2]):
            raise ValueError("Requested patch size is too big!")

        if "y" not in self.data["training"] and discard is not None:
            raise ValueError(
                "Cannot use a patch discard policy when RGB data is not loaded!"
            )

        for b in range(batch_size):
            bid = batch_id * batch_size + b
            if self.presample_epochs:
                if has_raw:
                    bx[b] = self.data["training"]["x"][bid].astype(
                        np.float) / (2 ** 16 - 1)
                if has_rgb:
                    by[b] = self.data["training"]["y"][bid].astype(
                        np.float) / (2 ** 8 - 1)
            else:
                current_rgb = self.data["training"]["y"][
                    bid] if has_rgb else None
                xx, yy = sample_patch(
                    current_rgb,
                    rgb_patch_size,
                    discard,
                    max_attempts,
                    self.train_image_shape_rgb,
                )
                rx, ry = xx // 2, yy // 2

                if has_raw:
                    current_raw = self.data["training"]["x"][bid]
                    bx[b] = current_raw[
                            ry: ry + raw_patch_size, rx: rx + raw_patch_size
                            ].astype(np.float) / (2 ** 16 - 1)

                if has_rgb:
                    by[b] = current_rgb[
                            yy: yy + rgb_patch_size, xx: xx + rgb_patch_size
                            ].astype(np.float) / (2 ** 8 - 1)

        if has_rgb and has_raw:
            return bx, by
        elif has_rgb:
            return by
        elif has_raw:
            return bx

    def next_validation_batch(self, batch_id, batch_size):
        """
        Return a validation batch.
        :param batch_id: integer from 0 to (#validation images // batch_size - 1)
        :param batch_size: integer, self explanatory
        :return: tuple of np arrays (RAW, RGB) or np array (RGB)
        """
        rgb_patch = self.valid_patch_size_rgb

        if (batch_id + 1) * batch_size > self.count_validation:
            raise ValueError(
                "Not enough images for the requested batch_id & batch_size"
            )

        has_raw = "x" in self._loaded_data
        has_rgb = "y" in self._loaded_data
        bx = (
            np.zeros((batch_size, rgb_patch // 2, rgb_patch // 2, 4),
                     dtype=np.float32)
            if has_raw
            else None
        )
        by = (
            np.zeros((batch_size, rgb_patch, rgb_patch, 3), dtype=np.float32)
            if has_rgb
            else None
        )

        for b in range(batch_size):
            if has_raw:
                bx[b] = self.data["validation"]["x"][
                            batch_id * batch_size + b].astype(
                    np.float
                ) / (2 ** 16 - 1)
            if has_rgb:
                by[b] = self.data["validation"]["y"][
                            batch_id * batch_size + b].astype(
                    np.float
                ) / (2 ** 8 - 1)

        if has_rgb and has_raw:
            return bx, by
        elif has_rgb:
            return by
        elif has_raw:
            return bx

    def next_calibration_batch(self, batch_id, batch_size):
        """
        Return a calibration batch.
        :param batch_id: integer from 0 to (#calibration images // batch_size - 1)
        :param batch_size: integer, self explanatory
        :return: tuple of np arrays (RAW, RGB) or np array (RGB)
        """
        rgb_patch = self.calib_patch_size_rgb

        if (batch_id + 1) * batch_size > self.count_calibration:
            raise ValueError(
                "Not enough images for the requested batch_id & batch_size"
            )

        has_raw = "x" in self._loaded_data
        has_rgb = "y" in self._loaded_data
        bx = (
            np.zeros((batch_size, rgb_patch // 2, rgb_patch // 2, 4),
                     dtype=np.float32)
            if has_raw
            else None
        )
        by = (
            np.zeros((batch_size, rgb_patch, rgb_patch, 3), dtype=np.float32)
            if has_rgb
            else None
        )

        for b in range(batch_size):
            if has_raw:
                bx[b] = self.data["calibration"]["x"][
                            batch_id * batch_size + b].astype(
                    np.float
                ) / (2 ** 16 - 1)
            if has_rgb:
                by[b] = self.data["calibration"]["y"][
                            batch_id * batch_size + b].astype(
                    np.float
                ) / (2 ** 8 - 1)

        if has_rgb and has_raw:
            return bx, by
        elif has_rgb:
            return by
        elif has_raw:
            return bx

    def is_raw_and_rgb(self):
        return len(self._loaded_data) == 2

    @property
    def valid_patch_size_rgb(self):
        if "y" in self._loaded_data:
            patch_size = self.data["validation"]["y"].shape[1]
        else:
            patch_size = 2 * self.data["validation"]["x"].shape[1]
        return patch_size

    @property
    def calib_patch_size_rgb(self):
        if "y" in self._loaded_data:
            patch_size = self.data["calibration"]["y"].shape[1]
        else:
            patch_size = 2 * self.data["calibration"]["x"].shape[1]
        return patch_size

    @property
    def train_image_shape_rgb(self):
        if "y" in self._loaded_data:
            patch_size = self.data["training"]["y"].shape[1:]
        else:
            shape = self.data["training"]["x"].shape
            patch_size = (2 * shape[1], 2 * shape[2], shape[3])
        return patch_size

    @property
    def count_training(self):
        key = self._loaded_data[0]
        return self.data["training"][key].shape[0]

    @property
    def count_validation(self):
        key = self._loaded_data[0]
        return self.data["validation"][key].shape[0]

    @property
    def count_calibration(self):
        key = self._loaded_data[0]
        return self.data["calibration"][key].shape[0]

    def __repr__(self):
        args = [
            f'"{self._data_directory}"',
            f'load="{self._loaded_data}"',
            f"n_images={self._counts[0]}",
            f"v_images={self._counts[1]}",
            f"c_images={self._counts[2]}",
            f"val_rgb_patch_size={self._counts[3]}",
            f"val_rgb_patch_size={self.valid_patch_size_rgb}",
            f'discard="{self._val_discard}"',
        ]
        return f'Dataset({", ".join(args)})'

    def shapes(self):
        stats = {
            "path": self._data_directory,
        }

        for k in self._loaded_data:
            stats["training/{}".format(k)] = self.data["training"][k].shape
            stats["validation/{}".format(k)] = self.data["validation"][k].shape
            stats["calibration/{}".format(k)] = self.data["calibration"][
                k].shape

        return stats

    @property
    def loaded_data(self):
        if self._loaded_data == "xy":
            db_type = "raw+rgb"
        elif self._loaded_data == "y":
            db_type = "rgb"
        elif self._loaded_data == "x":
            db_type = "raw"
        return db_type

    def summary(self):
        valid_label = "" if self._val_discard is None else f", {self._val_discard}"
        return (
            f"Dataset[{os.path.split(self._data_directory)[-1]},{self.loaded_data}]: "
            f"{self.count_training} train. images ({np.prod(self.train_image_shape_rgb[:2]) / 1e6:.1f} Mpx) "
            f"+ {self.count_validation} valid. patches ({self.valid_patch_size_rgb} px{valid_label})"
            f"+ {self.count_calibration} calib. patches ({self.calib_patch_size_rgb} px{valid_label})"
        )

    def details(self):
        label = [self.summary()]

        for k, l in zip("xy", ["RAW", "RGB"]):
            if k in self._loaded_data:
                label.append(
                    f'{l} -> training {self.data["training"][k].shape} + validation {self.data["validation"][k].shape} + calibration {self.data["calibration"][k].shape}'
                )

        return "\n".join(label)

    def preprocess_batch(self, batch, **kwargs):
        """
        Implement this method to return the processed batch and its labels.
        """
        return batch

    def get_training_generator(self, batch_size, patch_size, discard="flat",
                               **kwargs):
        """
        Get a generator for training data. Can be used to construct a data pipeline:

        dp = tf.data.Dataset.from_generator(lambda: data.get_training_generator(batch_size, rgb_patch_size, discard),
            output_types=len(self._loaded_data) * (tf.float32, ))
        """
        for batch_id in range(self.count_training // batch_size):
            batch = self.next_training_batch(
                batch_id, batch_size, patch_size, discard)
            images, labels = self.preprocess_batch(batch, **kwargs)
            yield images, labels

    def get_validation_generator(self, batch_size, **kwargs):
        """
        Get a generator for validation data. Can be used to construct a data pipeline:

        dp = tf.data.Dataset.from_generator(lambda: data.get_validation_generator(batch_size),
            output_types=len(self._loaded_data) * (tf.float32, ))
        """
        for batch_id in range(self.count_validation // batch_size):
            batch = self.next_validation_batch(batch_id, batch_size)
            images, labels = self.preprocess_batch(batch, **kwargs)
            yield images, labels

    def get_calibration_generator(self, batch_size, **kwargs):
        """
        Get a generator for calibration data. Can be used to construct a data pipeline:

        dp = tf.data.Dataset.from_generator(lambda: data.get_calibration_generator(batch_size),
            output_types=len(self._loaded_data) * (tf.float32, ))
        """
        for batch_id in range(self.count_calibration // batch_size):
            batch = self.next_calibration_batch(batch_id, batch_size)
            images, labels = self.preprocess_batch(batch, **kwargs)
            yield images, labels

    def get_training_pipeline(self, batch_size, rgb_patch_size,
                              discard="flat"):
        import tensorflow as tf

        types = (
            tf.float32, tf.float32) if self.is_raw_and_rgb() else tf.float32
        shapes = (
            (batch_size, rgb_patch_size // 2, rgb_patch_size // 2, 4),
            (batch_size, rgb_patch_size, rgb_patch_size, 3),
        )
        return tf.data.Dataset.from_generator(
            lambda: self.get_training_generator(batch_size, rgb_patch_size,
                                                discard),
            output_types=types,
            output_shapes=shapes,
        )

    def get_validation_pipeline(self, batch_size):
        import tensorflow as tf

        return tf.data.Dataset.from_generator(
            lambda: self.get_validation_generator(batch_size),
            output_types=len(self._loaded_data) * (tf.float32,),
        )

    def get_calibration_pipeline(self, batch_size):
        import tensorflow as tf

        return tf.data.Dataset.from_generator(
            lambda: self.get_calibration_generator(batch_size),
            output_types=len(self._loaded_data) * (tf.float32,),
        )