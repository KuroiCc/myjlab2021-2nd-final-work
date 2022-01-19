from typing import List, Tuple

import numpy as np
import json


def save_image_json(face_names: list, face_encoding: List[np.ndarray], file_path: str):
    """
    Save to JSON as shape like List[Tuple[face_name, face_encoding]]
    """
    if len(face_names) == len(face_encoding):
        raise ValueError('face_names and face_encoding must be the same length')
    face_encoding_list = map(lambda x: x.tolist(), face_encoding)
    with open(file_path, 'w') as f:
        json.dump(list(zip(face_names, face_encoding_list)), f)


def load_image_json(file_path: str) -> Tuple[list, list]:
    """
    Load JSON saved by func `save_image_json`
    return Tuple[face_name_list, face_encoding_list]
    """
    with open(file_path, 'r') as f:
        face_name_list, face_encoding_list = zip(*json.load(f))

    return face_name_list, face_encoding_list
