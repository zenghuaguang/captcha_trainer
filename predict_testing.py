#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import io

import PIL
import cv2
import numpy as np
import PIL.Image as PIL_Image
import tensorflow as tf
from importlib import import_module
from config import *
from constants import RunMode
from pretreatment import preprocessing
from framework import GraphOCR


def get_image_batch(img_bytes):

    def load_image(path_or_bytes):
        if isinstance(path_or_bytes, bytes):
            nparr = np.fromstring(path_or_bytes, np.uint8)
            im = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        else:
            im = cv2.imread(path_or_bytes, cv2.IMREAD_GRAYSCALE)
            # The OpenCV cannot handle gif format images, it will return None.
        if im is None:
            path_or_stream = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
            pil_image = PIL.Image.open(path_or_stream)
            rgb = pil_image.split()
            size = pil_image.size

            if len(rgb) > 3 and REPLACE_TRANSPARENT:
                background = PIL.Image.new('RGB', pil_image.size, (255, 255, 255))
                background.paste(pil_image, (0, 0, size[0], size[1]), pil_image)
            pil_image = background

            if IMAGE_CHANNEL == 1:
                pil_image = pil_image.convert('L')

            im = np.array(pil_image)
        im = preprocessing(im,GAMMA, BINARYZATION, SMOOTH, BLUR).astype(np.float32)
        if RESIZE[0] == -1:
            ratio = RESIZE[1] / size[1]
            resize_width = int(ratio * size[0])
            im = cv2.resize(im, (resize_width, RESIZE[1]))
        else:
            im = cv2.resize(im, (RESIZE[0], RESIZE[1]))
        im = im.swapaxes(0, 1)
        return (im[:, :, np.newaxis] if IMAGE_CHANNEL == 1 else im[:, :]) / 255.

    return [load_image(index) for index in [img_bytes]]


def decode_maps(charset):
    return {index: char for index, char in enumerate(charset, 0)}


def predict_func(image_batch, _sess, dense_decoded, op_input):
    dense_decoded_code = _sess.run(dense_decoded, feed_dict={
        op_input: image_batch,
    })
    decoded_expression = []
    for item in dense_decoded_code:
        expression = ''

        for char_index in item:
            if char_index == -1:
                expression += ''
            else:
                expression += decode_maps(GEN_CHAR_SET)[char_index]
        decoded_expression.append(expression)
    return ''.join(decoded_expression) if len(decoded_expression) > 1 else decoded_expression[0]


if __name__ == '__main__':

    if WARP_CTC:
        import_module('warpctc_tensorflow')
    graph = tf.Graph()
    tf_checkpoint = tf.train.latest_checkpoint(MODEL_PATH)
    sess = tf.Session(
        graph=graph,
        config=tf.ConfigProto(
            # allow_soft_placement=True,
            # log_device_placement=True,
            gpu_options=tf.GPUOptions(
                allocator_type='BFC',
                # allow_growth=True,  # it will cause fragmentation.
                per_process_gpu_memory_fraction=0.01
            ))
    )
    graph_def = graph.as_graph_def()

    with graph.as_default():
        sess.run(tf.global_variables_initializer())
        # with tf.gfile.GFile(COMPILE_MODEL_PATH.replace('.pb', '_{}.pb'.format(int(0.95 * 10000))), "rb") as f:
        #     graph_def_file = f.read()
        # graph_def.ParseFromString(graph_def_file)
        # print('{}.meta'.format(tf_checkpoint))
        model = GraphOCR(
            RunMode.Predict,
            NETWORK_MAP[NEU_CNN],
            NETWORK_MAP[NEU_RECURRENT]
        )
        model.build_graph()
        saver = tf.train.Saver(tf.global_variables())

        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
        _ = tf.import_graph_def(graph_def, name="")

    dense_decoded_op = sess.graph.get_tensor_by_name("dense_decoded:0")
    x_op = sess.graph.get_tensor_by_name('input:0')
    sess.graph.finalize()

    # Fill in your own sample path
    image_dir = r"D:\workspace\Train\captcha3"
    right_num = 0
    for i, p in enumerate(os.listdir(image_dir)):
        n = os.path.join(image_dir, p)
        if i > 1000:
            break
        with open(n, "rb") as f:
            b = f.read()

        batch = get_image_batch(b)
        predict_text = predict_func(
            batch,
            sess,
            dense_decoded_op,
            x_op,
        )
        if(re.search(TRAINS_REGEX, p).group()==predict_text):
            right_num+=1
        else:
            print(i, p, predict_text)

    print(right_num,'/',1000,"end")

