import io

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

    def _transform_bytes_image_to_RGB_ndarray(self, bytes):
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

        image = np.array(image)
        return image

    def get_best_match_from_file(self, file_path: str):
        """
        ファイルから最も似てる顔を返す
        返すデータは
        name, similarity, src_url, comment
        """
        image = face_recognition.load_image_file(file_path)
        return self.get_best_match_from_RGB_ndarray(image)

    def get_best_match_from_bytes(self, bytes):
        image = self._transform_bytes_image_to_RGB_ndarray(bytes)
        return self.get_best_match_from_RGB_ndarray(image)

    def get_best_match_from_RGB_ndarray(self, RGB_ndarray: np.ndarray):
        face_location = self.find_face(RGB_ndarray)
        if face_location is None:
            return
        face_encoding = face_recognition.face_encodings(RGB_ndarray, [face_location])

        if face_encoding:
            face_encoding = face_encoding[0]
            face_distances = face_recognition.face_distance(self.model, face_encoding)

            best_match_index = np.argmin(face_distances)
            res = self.model_data[best_match_index]
            return (
                res.name,
                1 - face_distances[best_match_index],
                res.src_url,
                res.comment,
            )

    def find_face(self, RGB_ndarray: np.ndarray):
        gray = cv2.cvtColor(RGB_ndarray[:, :, ::-1], cv2.COLOR_BGR2GRAY)
        print(RGB_ndarray.shape)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
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
