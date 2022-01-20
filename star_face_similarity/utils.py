import os
import csv
import base64
from typing import List, Tuple

import json
import requests
import numpy as np
from pydantic import BaseModel


class CsvFaceModel(BaseModel):
    name: str
    src_url: str = ''
    comment: str = ''
    encoding: np.ndarray

    class Config:
        arbitrary_types_allowed = True


def save_data_csv(rows: List[CsvFaceModel], file_path: str):
    """
    Save to a CSV file.
    It will tansfor CsvFaceModel.encoding(np.ndarray) to base64
    """

    def encode_row(row: CsvFaceModel):
        dic = row.dict()
        dic['base64_encoding'] = \
            base64.b64encode(dic['encoding'].tobytes()).decode("utf-8")

        dic.pop('encoding')
        return dic

    with open(file_path, 'w', newline='') as f:
        csv_writer = csv.DictWriter(
            f, fieldnames=['name', 'src_url', 'comment', 'base64_encoding']
        )
        csv_writer.writeheader()
        csv_writer.writerows(list(map(encode_row, rows)))


def load_data_csv(file_path):
    """
    Load CSV save by func `save_data_csv`
    """

    def decode_row(row: dict):
        row['encoding'] = \
            np.frombuffer(base64.b64decode(row['base64_encoding']), dtype="float64")
        row.pop('base64_encoding')
        return CsvFaceModel(**row)

    with open(file_path, newline='') as f:
        csv_reader = csv.DictReader(f)
        data = list(map(decode_row, csv_reader))

    return data


def crawling_image(name: str, image_url: str, path: str = 'images'):
    res = requests.get(image_url)
    if not os.path.exists(path):
        os.makedirs(os.path.abspath(path))
    extension = image_url.rsplit('.')[-1]
    with open(f'{path}/{name}.{extension}', 'wb') as f:
        f.write(res.content)

    return os.path.abspath(f'{path}/{name}.{extension}')


def find_all_file(path: str):
    file_list = []
    files = os.listdir(path)

    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            inner_file_list = find_all_file(file_path)
            file_list = file_list + inner_file_list
        elif os.path.isfile(file_path):
            file_list.append(file_path)

    return file_list


def save_image_json(face_names: list, face_encoding: List[np.ndarray], file_path: str):
    """
    ! deprecated
    Save to JSON as shape like List[Tuple[face_name, face_encoding]]
    """
    print('This function is deprecated. Use save_data_csv instead')
    if len(face_names) == len(face_encoding):
        raise ValueError('face_names and face_encoding must be the same length')
    face_encoding_list = map(lambda x: x.tolist(), face_encoding)
    with open(file_path, 'w') as f:
        json.dump(list(zip(face_names, face_encoding_list)), f)


def load_image_json(file_path: str) -> Tuple[list, list]:
    """
    ! deprecated
    Load JSON saved by func `save_image_json`
    return Tuple[face_name_list, face_encoding_list]
    """
    print('This function is deprecated. Use load_data_csv instead')
    with open(file_path, 'r') as f:
        face_name_list, face_encoding_list = zip(*json.load(f))

    return face_name_list, face_encoding_list
