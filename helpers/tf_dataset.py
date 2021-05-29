# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)


# Standard libraries
import os

# External Libraries
import tensorflow as tf
from loguru import logger

# Internal Libraries
from helpers import loading
from helpers.loading import tf_sample_patch


class Dataset(object):
    """Provides a Dataset class that loads full resolution training images
    and samples from them randomly."""

    def __init__(
            self,
            data_dir=None,
            *,
            seed=2468,
            load="xy",
            presample_epochs=0,
            n_images=120,
            preloaded_rgb_train_data=None,
            train_rgb_patch_size=0,
            train_n_patches=1,
            v_images=30,
            preloaded_rgb_val_data=None,
            val_rgb_patch_size=128,
            val_n_patches=1,
            val_discard="flat-aggressive",
            c_images=50,
            batch_size=64,
            calibrate=False,
            **kwargs
    ):

        # Arguments verification
        if not any(load == allowed for allowed in {"xy", "x", "y"}):
            raise ValueError("Invalid X/Y data requested!")

        if train_rgb_patch_size == 0:
            train_rgb_patch_size = val_rgb_patch_size
        elif presample_epochs == 0:
            raise ValueError(
                "Setting training patch size here only works when preloading the patches "
                "(when presample_epochs>0). Otherwise, full resolution images are loaded."
            )

        if data_dir and not os.path.isdir(data_dir):
            if "/" in data_dir or "\\" in data_dir:
                raise ValueError(
                    f"Cannot find the data directory: {data_dir}")

            if os.path.isdir(
                    os.path.join("data/raw/training_data/", data_dir)):
                data_dir = os.path.join("data/raw/training_data/",
                                        data_dir)
            elif os.path.isdir(os.path.join("data/rgb/", data_dir)):
                data_dir = os.path.join("data/rgb/", data_dir)
            else:
                raise ValueError(
                    f"Cannot find the data directory: {data_dir}")

        if presample_epochs > 0:
            logger.info(
                "Sampling {} patches per training image beforehand.".format(
                    presample_epochs))
            train_n_patches = presample_epochs

        # Initializations
        c_images = c_images if calibrate else 0
        self.data = {"training": {}, "validation": {}, "calibration": {}}
        self.files = {}
        self.batch_size = batch_size
        self._loaded_data = load
        self._data_directory = data_dir
        self._counts = (
            n_images * train_n_patches,
            v_images * val_n_patches,
            c_images,
            val_n_patches,
        )
        self._val_discard = val_discard
        if "y" in load:
            self.train_image_shape_rgb = (
                train_rgb_patch_size, train_rgb_patch_size, 3)
        else:
            self.train_image_shape_rgb = (
                2 * train_rgb_patch_size, 2 * train_rgb_patch_size, 4)
        self.val_patch_size_rgb = val_rgb_patch_size

        if data_dir:
            # Discover images files to sample from.
            self.files["training"], self.files["validation"], self.files[
                "calibration"] = loading.discover_images(
                data_dir, randomize=seed, n_images=n_images,
                v_images=v_images, c_images=c_images,
            )
        else:
            logger.info("No data directory given.")

        # flags for storing way train data was prepared, with default values.
        self.preloading_train = 0

        # Preparing training data
        if preloaded_rgb_train_data is not None:
            self.data["training"]['y'] = tf.math.divide(
                preloaded_rgb_train_data, (2 ** 8 - 1))
            self.batched_data_train = tf.data.Dataset.from_tensor_slices(
                self.data["training"]['y']).batch(batch_size,
                                                  drop_remainder=True)
            self.preloading_train = 1

        elif presample_epochs != 0:
            self.data["training"] = loading.load_patches(
                self.files["training"],
                data_dir,
                patch_size=train_rgb_patch_size // 2,
                n_patches=train_n_patches,
                load=load,
                discard=val_discard,
            )
            self.preloading_train = 1
        else:
            self.data["training"] = loading.load_images(
                self.files["training"], data_dir, load=load
            )

        # Prepare validation data
        if preloaded_rgb_val_data is not None:
            self.data["validation"]['y'] = tf.math.divide(
                preloaded_rgb_val_data, (2 ** 8 - 1))
        else:
            self.data["validation"] = loading.load_patches(
                self.files["validation"],
                data_dir,
                patch_size=val_rgb_patch_size // 2,
                n_patches=val_n_patches,
                load=load,
                discard=val_discard,
            )

        # Prepare calibration data. Only one method available, similar to val.
        if calibrate:
            self.data["calibration"] = loading.load_patches(
                self.files["calibration"],
                data_dir,
                patch_size=val_rgb_patch_size // 2,
                n_patches=val_n_patches,
                load=load,
                discard=val_discard,
            )

        # Conversion to tensor and batching of loaded data
        if "x" in load:
            if self.preloading_train:
                self.data["training"][
                    "x"] = tf.data.Dataset.from_tensor_slices(
                    self.data["training"]["x"]).batch(batch_size,
                                                      drop_remainder=True)
            else:
                self.data["training"][
                    "x"] = tf.convert_to_tensor(self.data["training"]["x"])
            self.data["validation"]["x"] = tf.data.Dataset.from_tensor_slices(
                self.data["validation"]["x"]).batch(batch_size,
                                                    drop_remainder=True)
            if calibrate:
                self.data["calibration"][
                    "x"] = tf.data.Dataset.from_tensor_slices(
                    self.data["calibration"]["x"]).batch(batch_size,
                                                         drop_remainder=True)
        if "y" in load:
            if self.preloading_train:
                self.data["training"][
                    "y"] = tf.data.Dataset.from_tensor_slices(
                    self.data["training"]["y"]).batch(batch_size,
                                                      drop_remainder=True)
            else:
                self.data["training"][
                    "y"] = tf.convert_to_tensor(self.data["training"]["y"])
            self.data["validation"]["y"] = tf.data.Dataset.from_tensor_slices(
                self.data["validation"]["y"]).batch(batch_size,
                                                    drop_remainder=True)
            if calibrate:
                self.data["calibration"]["y"] = tf.data.Dataset.from_tensor_slices(
                    self.data["calibration"]["y"]).batch(batch_size,
                                                         drop_remainder=True)

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

        # Allocate memory for the batch
        has_rgb = "y" in self._loaded_data
        by = (
            tf.zeros((batch_size, rgb_patch_size, rgb_patch_size, 3),
                     dtype=tf.float32)
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
            current_rgb = self.data["training"]["y"][bid]
            xx, yy = tf_sample_patch(
                current_rgb,
                rgb_patch_size,
                discard,
                max_attempts,
                self.train_image_shape_rgb,
            )
            by[b] = tf.cast(current_rgb[
                            yy: yy + rgb_patch_size,
                            xx: xx + rgb_patch_size
                            ], tf.float32) / (2 ** 8 - 1)

        return by

    def is_raw_and_rgb(self):
        return len(self._loaded_data) == 2

    @property
    def valid_patch_size_rgb(self):
        if "y" in self._loaded_data:
            patch_size = self.val_patch_size_rgb
        else:
            patch_size = 2 * self.val_patch_size_rgb
        return patch_size

    @property
    def calib_patch_size_rgb(self):
        if "y" in self._loaded_data:
            patch_size = self.val_patch_size_rgb
        else:
            patch_size = 2 * self.val_patch_size_rgb
        return patch_size

    @property
    def count_training(self):
        return self._counts[0]

    @property
    def count_validation(self):
        return self._counts[1]

    @property
    def count_calibration(self):
        return self._counts[2]

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
            f"{self.count_training} train. images ({tf.prod(self.train_image_shape_rgb[:2]) / 1e6:.1f} Mpx) "
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

    def get_training_generator(self, discard="flat", **kwargs):
        """
        Get a generator for training data. Can be used to construct a data pipeline:

        dp = tf.data.Dataset.from_generator(lambda: data.get_training_generator(batch_size, rgb_patch_size, discard),
            output_types=len(self._loaded_data) * (tf.float32, ))
        """
        if self.preloading_train:
            for batch in self.batched_data_train:
                images, labels = self.preprocess_batch(batch, **kwargs)
                yield images, labels
        else:
            for batch_id in range(self.count_training // self.batch_size):
                batch = self.next_training_batch(batch_id, discard)
                images, labels = self.preprocess_batch(batch, **kwargs)
                yield images, labels

    def get_validation_generator(self, **kwargs):
        """
        Get a generator for validation data. Can be used to construct a data pipeline:

        dp = tf.data.Dataset.from_generator(lambda: data.get_validation_generator(batch_size),
            output_types=len(self._loaded_data) * (tf.float32, ))
        """
        for batch in self.data["validation"]["y"]:
            images, labels = self.preprocess_batch(batch, **kwargs)
            yield images, labels

    def get_calibration_generator(self, **kwargs):
        """
        Get a generator for calibration data. Can be used to construct a data pipeline:

        dp = tf.data.Dataset.from_generator(lambda: data.get_calibration_generator(batch_size),
            output_types=len(self._loaded_data) * (tf.float32, ))
        """
        for batch in self.data["calibration"]["y"]:
            images, labels = self.preprocess_batch(batch, **kwargs)
            yield images, labels

    def get_training_pipeline(self, discard="flat"):
        types = (
            tf.float32, tf.float32) if self.is_raw_and_rgb() else tf.float32
        shapes = (
            (self.batch_size, self.train_image_shape_rgb[0] // 2,
             self.train_image_shape_rgb[1] // 2, 4),
            (self.batch_size, self.train_image_shape_rgb[0],
             self.train_image_shape_rgb[1], 3),
        )
        return tf.data.Dataset.from_generator(
            lambda: self.get_training_generator(discard),
            output_types=types,
            output_shapes=shapes,
        )

    def get_validation_pipeline(self):
        return tf.data.Dataset.from_generator(
            lambda: self.get_validation_generator(),
            output_types=len(self._loaded_data) * (tf.float32,),
        )

    def get_calibration_pipeline(self):
        return tf.data.Dataset.from_generator(
            lambda: self.get_calibration_generator(),
            output_types=len(self._loaded_data) * (tf.float32,),
        )
