import cv2
import numpy as np

from api.detector import Detector


class IdentityCardDetector(Detector):
    def __init__(self, sess):
        super().__init__(sess)

    def id_number_detect(self, img):
        id_number_line_img = self.__id_number_line_detect(img)
        id_number_img = self.__find_id_number(id_number_line_img)
        return id_number_img

    def __id_number_line_detect(self, img):
        id_number_line_img = None
        boxes = super().detect(img)
        boxes = np.asarray(boxes, np.int32)

        for i in range(boxes.shape[0]):
            if (boxes[i][2] - - boxes[i][0]) / (boxes[i][5] - boxes[i][1]) > 18 and \
                                    (boxes[i][2] - - boxes[i][0]) / img.shape[1] > 0.8:
                id_number_line_img = img[boxes[i][1]:boxes[i][5], boxes[i][0]:boxes[i][2], :]
                break

        if id_number_line_img is not None:
            p = 32 / id_number_line_img.shape[0]
            height = int(id_number_line_img.shape[0] * p)
            width = int(id_number_line_img.shape[1] * p)
            id_number_line_img = cv2.resize(id_number_line_img, (width, height), interpolation=cv2.INTER_CUBIC)

        return id_number_line_img

    def __find_id_number(self, id_number_line_img):
        gray = cv2.cvtColor(id_number_line_img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        p = 0.32
        x = int(w * p)
        y = 0
        i = gray[y:y + h, x:x + w]
        return i
