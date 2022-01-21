import logging
import unittest
from datetime import datetime

import cv2
import requests
import numpy as np

from star_face_similarity import StarFaceSimilarity


class TestSFS(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.sfs = StarFaceSimilarity('./app/data.csv', './app/haarcascade_frontalface_alt2.xml')

        print("===================================================")
        logging.basicConfig(level=logging.DEBUG)

    def setUp(self):
        meg = f"---{self._testMethodName}---"
        print(meg, end="")
        print("-" * (70 - len(meg)))
        self.time_begin = datetime.now()

    def tearDown(self):
        t = datetime.now() - self.time_begin
        print(f"\nin: {t.total_seconds():.3f}s", end="\n\n\n")

    def test_get_best_match_from_file(self):
        t = self.sfs.get_best_match_from_file('./temp/images/sei.jpg')
        self.assertEqual(self.sfs.model_data[t[0]].name, '神木隆之介')

    def test_get_best_match_from_bytes(self):
        url = 'https://img.ranking.net/uploads/item/image/66/80/b6/default_2000030327.jpg'
        res = requests.get(url)
        t = self.sfs.get_best_match_from_bytes(res.content)
        self.assertEqual(self.sfs.model_data[t[0]].name, '新垣結衣')

    def test_find_one_face_cv2(self):
        img = cv2.imread('./temp/images/sei.jpg')
        img = img[:, :, ::-1]
        res = self.sfs.find_one_face_cv2(img)
        self.assertTrue(res)

    def test_find_one_face(self):
        img = cv2.imread('./temp/images/sei.jpg')
        img = img[:, :, ::-1]
        res = self.sfs.find_one_face(img)
        self.assertTrue(res)

    def test_surround_face(self):
        img = cv2.imread('./temp/images/sei.jpg')
        img2 = cv2.imread('./temp/images/surround_face.png')
        img = img[:, :, ::-1]
        face_location = self.sfs.find_one_face(img)
        img = self.sfs.surround_face(img, face_location, 3)
        # cv2.imwrite('./tests/surround_face.png', img[:, :, ::-1])
        self.assertTrue(np.array_equal(img, img2[:, :, ::-1]))


if __name__ == "__main__":
    unittest.main(verbosity=0)
