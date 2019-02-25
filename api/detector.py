import os

import cv2
import numpy as np
import tensorflow as tf
from core.rpn_msr.proposal_layer_tf import proposal_layer
from tensorflow.python.platform import gfile
from core.text_connector.detectors import TextDetector
from core.text_connector.text_connect_cfg import Config as TextLineCfg

from core.fast_rcnn.config import cfg, cfg_from_file


class Detector:
    def __init__(self, sess):
        self.sess = sess
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'conf.yml')
        print(config_file)
        cfg_from_file(config_file)
        input_img, output_cls_prob, output_box_pred = self.__load_model(sess)
        self.input_img = input_img
        self.output_cls_prob = output_cls_prob
        self.output_box_pred = output_box_pred

    def __load_model(self, sess):
        pb_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ctpn.pb')
        with gfile.FastGFile(pb_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        sess.run(tf.global_variables_initializer())

        input_img = sess.graph.get_tensor_by_name('Placeholder:0')
        output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
        output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')
        return input_img, output_cls_prob, output_box_pred

    def __resize_im(self, im, scale, max_scale=None):
        f = float(scale) / min(im.shape[0], im.shape[1])
        if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
            f = float(max_scale) / max(im.shape[0], im.shape[1])
        return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

    def __get_blobs(self, im, rois):
        blobs = {'data': None, 'rois': None}
        blobs['data'], im_scale_factors = self.__get_image_blob(im)
        return blobs, im_scale_factors

    def __get_image_blob(self, im):
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = self.__im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def __im_list_to_blob(self, ims):
        """Convert a list of images into a network input.

        Assumes images are already prepared (means subtracted, BGR order, ...).
        """
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        num_images = len(ims)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                        dtype=np.float32)
        for i in range(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

        return blob

    def detect(self, img):
        """
        查找文字轮廓
        :param img: 图片数组
        :return:  boxes[]——[左上角x,左上角y,右上角x,右上角y,左下角x,左下角y，右下角x,右下角y]
        多个boxes之间，默认以从上到下的顺序排序（左上角y坐标）
        """
        img, scale = self.__resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        blobs, im_scales = self.__get_blobs(img, None)
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
        cls_prob, box_pred = self.sess.run([self.output_cls_prob, self.output_box_pred],
                                      feed_dict={self.input_img: blobs['data']})
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)
        scores = rois[:, 0]
        boxes = rois[:, 1:5] / im_scales[0]
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])

        temp = boxes / scale
        temp = np.asarray(temp, np.int32)  # 转为int
        # 排序
        index = np.argsort(temp[:, 1], 0)
        res = np.zeros(temp.shape, np.int32)
        for i in range(res.shape[0]):
            res[i] = temp[index[i]]

        return res

#
# if __name__ == '__main__':
#     img = cv2.imread('001.jpg')
#     with tf.Session() as sess:
#         d = Detector(sess)
#         s = d.detect(img)
#         print(s)
