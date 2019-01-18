import cv2
import tensorflow as tf

from api.identity_card_detector import IdentityCardDetector

if __name__ == '__main__':
    img = cv2.imread('002.jpg')
    sess = tf.Session()
    d = IdentityCardDetector(sess)
    ii = d.id_number_detect(img)
    cv2.imshow('t', ii)
    cv2.waitKey(0)
