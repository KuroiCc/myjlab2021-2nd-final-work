import io
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import face_recognition

from star_face_similarity.utils import load_data_csv, rotateImage


class StarFaceSimilarity:

    def __init__(self, model_data_path: str, faceCascade_path: str) -> None:
        self.model_data = load_data_csv(model_data_path)
        self.model = [i.encoding for i in self.model_data]

        self.faceCascade = cv2.CascadeClassifier(faceCascade_path)

    def get_rgb_ndarray_img_from_bytes(self, bytes):
        image = Image.open(io.BytesIO(bytes))

        try:
            # exif情報取得
            exifinfo = image._getexif()
            # exif情報からOrientationの取得
            orientation = exifinfo.get(0x112, 1)
            # 画像を回転
            image = rotateImage(image, orientation)
        except Exception as e:
            print(e)

        return np.array(image)

    def find_one_face(self, rgb_ndarray_img: np.ndarray):
        face_locations = face_recognition.face_locations(rgb_ndarray_img)

        if not face_locations:
            return None

        # find the largest one
        area_list = [(r - l) * (b - t) for t, r, b, l in face_locations]

        return face_locations[area_list.index(max(area_list))]

    def find_one_face_cv2(self, rgb_ndarray_img: np.ndarray):
        gray = cv2.cvtColor(rgb_ndarray_img[:, :, ::-1], cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(20, 20),
        )

        if type(faces) != np.ndarray:
            return None

        # find the largest one
        area_list = [w * h for _, _, w, h in faces]

        # transform (x, y, w, h) to (top, right, bottom, left)
        x, y, w, h = faces[area_list.index(max(area_list))]
        # return faces
        return y, x + w, y + h, x

    def get_best_match(
        self,
        rgb_ndarray_img: np.ndarray,
        face_location: Optional[Tuple[float, float, float, float]] = None,
    ):
        if face_location is None:
            face_location = self.find_one_face(rgb_ndarray_img)

        if face_location is None:
            return None

        face_encoding = \
            face_recognition.face_encodings(rgb_ndarray_img, [face_location])[0]

        face_distances = face_recognition.face_distance(self.model, face_encoding)
        best_match_index = np.argmin(face_distances)

        return best_match_index, 1 - face_distances[best_match_index]

    def get_best_match_from_file(
        self,
        file_path: str,
        face_location: Optional[Tuple[float, float, float, float]] = None,
    ):
        """
        ファイルから最も似てる顔を返す
        """
        image = face_recognition.load_image_file(file_path)
        return self.get_best_match(image, face_location)

    def get_best_match_from_bytes(
        self,
        bytes,
        face_location: Optional[Tuple[float, float, float, float]] = None,
    ):
        image = self.get_rgb_ndarray_img_from_bytes(bytes)
        return self.get_best_match(image, face_location)

    def surround_face(
        self,
        rgb_ndarray_img: np.ndarray,
        face_location: Optional[Tuple[float, float, float, float]],
        thickness=3,
    ):
        top, right, bottom, left = face_location
        rgb_ndarray_img = rgb_ndarray_img[:, :, ::-1]
        cv2.rectangle(rgb_ndarray_img, (left, top), (right, bottom), (0, 0, 255), thickness)

        return rgb_ndarray_img[:, :, ::-1]
