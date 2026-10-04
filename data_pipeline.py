import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


class TinyImageNetData:
    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.autotune = tf.data.AUTOTUNE
        self.mean = tf.constant([0.485, 0.456, 0.406], tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], tf.float32)
        self.class_to_index, self.class_to_names = self._load_classes()

    def _load_classes(self):
        with (self.data_dir / "wnids.txt").open() as handle:
            class_names = [line.strip() for line in handle if line.strip()]
        class_to_index = {name: index for index, name in enumerate(class_names)}
        class_to_names = {}
        with (self.data_dir / "words.txt").open(encoding="utf-8") as handle:
            for line in handle:
                key, value = line.rstrip("\n").split("\t", 1)
                class_to_names[key] = value
        return class_to_index, class_to_names

    def _train_files(self) -> Tuple[np.ndarray, np.ndarray]:
        paths, labels = [], []
        train_dir = self.data_dir / "train"
        for class_name, label in self.class_to_index.items():
            image_dir = train_dir / class_name / "images"
            for path in image_dir.glob("*.JPEG"):
                paths.append(str(path))
                labels.append(label)
        return np.asarray(paths), np.asarray(labels, dtype=np.int32)

    def _validation_files(self) -> Tuple[np.ndarray, np.ndarray]:
        paths, labels = [], []
        val_dir = self.data_dir / "val"
        with (val_dir / "val_annotations.txt").open() as handle:
            annotations = [line.strip().split("\t") for line in handle if line.strip()]
        for filename, class_name, *_ in annotations:
            paths.append(str(val_dir / "images" / filename))
            labels.append(self.class_to_index[class_name])
        return np.asarray(paths), np.asarray(labels, dtype=np.int32)

    def _decode(self, path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.config.image_size, self.config.image_size])
        return tf.cast(image, tf.float32) / 255.0, label

    def _cutout(self, images, size, probability):
        batch_size = tf.shape(images)[0]
        height, width, channels = tf.shape(images)[1], tf.shape(images)[2], tf.shape(images)[3]
        apply = tf.random.uniform([batch_size]) < probability
        centers_x = tf.random.uniform([batch_size], 0, tf.cast(width, tf.float32))
        centers_y = tf.random.uniform([batch_size], 0, tf.cast(height, tf.float32))
        x1 = tf.cast(tf.maximum(0.0, centers_x - size), tf.int32)
        y1 = tf.cast(tf.maximum(0.0, centers_y - size), tf.int32)
        x2 = tf.cast(tf.minimum(tf.cast(width, tf.float32), centers_x + size), tf.int32)
        y2 = tf.cast(tf.minimum(tf.cast(height, tf.float32), centers_y + size), tf.int32)

        def apply_one(index):
            image = images[index]
            zeros = tf.zeros([y2[index] - y1[index], x2[index] - x1[index], channels], image.dtype)
            mask = tf.pad(zeros, [[y1[index], height - y2[index]], [x1[index], width - x2[index]], [0, 0]], constant_values=1)
            return tf.where(apply[index], image * mask, image)

        return tf.map_fn(apply_one, tf.range(batch_size), fn_output_signature=images.dtype)

    def _preprocess_train(self, images, labels):
        image_size = self.config.image_size
        images = tf.image.resize_with_crop_or_pad(images, image_size + 8, image_size + 8)
        images = tf.image.random_crop(images, [tf.shape(images)[0], image_size, image_size, 3])
        images = tf.image.random_flip_left_right(images)
        augmentation = self.config.augmentation
        if augmentation["jitter_prob"] > 0:
            apply = tf.random.uniform([tf.shape(images)[0], 1, 1, 1]) < augmentation["jitter_prob"]
            strength = augmentation["jitter_strength"]
            jittered = tf.image.random_brightness(images, 0.8 * strength)
            jittered = tf.image.random_contrast(jittered, 1.0 - 0.8 * strength, 1.0 + 0.8 * strength)
            jittered = tf.image.random_saturation(jittered, 1.0 - 0.8 * strength, 1.0 + 0.8 * strength)
            jittered = tf.image.random_hue(jittered, 0.2 * strength)
            images = tf.where(apply, tf.clip_by_value(jittered, 0.0, 1.0), images)
        if augmentation["cutout_prob"] > 0:
            images = self._cutout(images, augmentation["cutout_hw"], augmentation["cutout_prob"])
        return self._normalize(images), labels

    def _normalize(self, images):
        if self.config.augmentation["normalize"] == "meanstd":
            images = (images - self.mean) / self.std
        if self.config.augmentation["fp16"] and tf.config.list_physical_devices("GPU"):
            images = tf.cast(images, tf.float16)
        return images

    def _mixup(self, images, labels, alpha):
        batch_size = tf.shape(images)[0]
        first = tf.random.gamma([batch_size], alpha)
        second = tf.random.gamma([batch_size], alpha)
        lam = first / (first + second)
        lam_images = tf.cast(tf.reshape(lam, [batch_size, 1, 1, 1]), images.dtype)
        lam_labels = tf.cast(tf.reshape(lam, [batch_size, 1]), labels.dtype)
        indices = tf.random.shuffle(tf.range(batch_size))
        return (images * lam_images + tf.gather(images, indices) * (1.0 - lam_images), labels * lam_labels + tf.gather(labels, indices) * (1.0 - lam_labels))

    def _cutmix(self, images, labels, alpha):
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        first = tf.random.gamma([batch_size], alpha)
        second = tf.random.gamma([batch_size], alpha)
        lam = first / (first + second)
        cut_ratio = tf.sqrt(1.0 - lam)
        cut_width = tf.cast(tf.cast(width, tf.float32) * cut_ratio, tf.int32)
        cut_height = tf.cast(tf.cast(height, tf.float32) * cut_ratio, tf.int32)
        centers_x = tf.random.uniform([batch_size], 0, width, dtype=tf.int32)
        centers_y = tf.random.uniform([batch_size], 0, height, dtype=tf.int32)
        x1 = tf.clip_by_value(centers_x - cut_width // 2, 0, width)
        y1 = tf.clip_by_value(centers_y - cut_height // 2, 0, height)
        x2 = tf.clip_by_value(centers_x + cut_width // 2, 0, width)
        y2 = tf.clip_by_value(centers_y + cut_height // 2, 0, height)
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)

        def apply_one(index):
            zeros = tf.zeros([y2[index] - y1[index], x2[index] - x1[index], 3], images.dtype)
            mask = tf.pad(zeros, [[y1[index], height - y2[index]], [x1[index], width - x2[index]], [0, 0]], constant_values=1)
            mixed_image = images[index] * mask + shuffled_images[index] * (1.0 - mask)
            replaced_area = tf.cast((x2[index] - x1[index]) * (y2[index] - y1[index]), labels.dtype)
            total_area = tf.cast(height * width, labels.dtype)
            adjusted_lam = 1.0 - replaced_area / total_area
            mixed_label = labels[index] * adjusted_lam + shuffled_labels[index] * (1.0 - adjusted_lam)
            return mixed_image, mixed_label

        return tf.map_fn(apply_one, tf.range(batch_size), fn_output_signature=(images.dtype, labels.dtype))

    def _mixup_cutmix(self, images, labels):
        choice = tf.random.uniform([])
        if self.config.mixup_alpha > 0:
            images, labels = tf.cond(choice < self.config.mixup_probability, lambda: self._mixup(images, labels, self.config.mixup_alpha), lambda: (images, labels))
        if self.config.cutmix_alpha > 0:
            images, labels = tf.cond(choice > 1.0 - self.config.cutmix_probability, lambda: self._cutmix(images, labels, self.config.cutmix_alpha), lambda: (images, labels))
        return images, labels

    def _dataset(self, paths, labels, train):
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels)).map(self._decode, num_parallel_calls=self.autotune)
        dataset = dataset.cache()
        if train:
            dataset = dataset.shuffle(5000)
        dataset = dataset.batch(self.config.batch_size, drop_remainder=train)
        if train:
            dataset = dataset.map(self._preprocess_train, num_parallel_calls=self.autotune)
        else:
            dataset = dataset.map(lambda x, y: (self._normalize(x), y), num_parallel_calls=self.autotune)
        dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, self.config.num_classes)), num_parallel_calls=self.autotune)
        if train:
            dataset = dataset.map(self._mixup_cutmix, num_parallel_calls=self.autotune)
        return dataset.prefetch(self.autotune)

    def datasets(self):
        train_paths, train_labels = self._train_files()
        rng = np.random.default_rng(self.config.seed)
        order = rng.permutation(len(train_paths))
        val_paths, val_labels = self._validation_files()
        return self._dataset(train_paths[order], train_labels[order], True), self._dataset(val_paths, val_labels, False)
