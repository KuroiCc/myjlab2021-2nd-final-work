import numpy as np
import face_recognition

from star_face_similarity.utils import load_data_csv


class StarFaceSimilarity:

    def __init__(self, model_data_path: str) -> None:
        self.model_data = load_data_csv(model_data_path)
        self.model = [i.encoding for i in self.model_data]

    def get_best_match_from_file(self, file_path: str):
        """
        ファイルから最も似てる顔を返す
        返すデータは
        name, similarity, src_url, comment
        """
        image_file = face_recognition.load_image_file(file_path)
        face_encoding = face_recognition.face_encodings(image_file)
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
