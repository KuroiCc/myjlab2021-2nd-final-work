import logging
import os

from PIL import Image
from pywebio import start_server
from pywebio.input import file_upload
from pywebio.output import (put_column, put_html, put_image, put_markdown, put_row, use_scope)
from pywebio.session import info
from star_face_similarity import StarFaceSimilarity

from app.config import config


def main():
    logger = logging.getLogger('main_app')
    model_data_path = config.MODEL_DATA_PATH.resolve().__str__()
    faceCascade_path = config.FACE_CASCADE_PATH.resolve().__str__()
    sfs = StarFaceSimilarity(model_data_path, faceCascade_path)

    put_markdown('# 有名人と顔の類似度\nどの有名人と一番似ているかを比較します。')

    img = file_upload("Select a image:", accept="image/*", required=True)
    print(f'Accept pictures: {img["filename"]}')
    with use_scope('res', clear=True):
        with open(config.LOADING_GIF_PATH, 'rb') as f:
            loading = f.read()

        put_image(loading, width='300px')

    nd_img = sfs.get_rgb_ndarray_img_from_bytes(img['content'])
    logger.info(f'{info.user_ip} Received image, start processing...')
    face_location = sfs.find_one_face(nd_img)
    if face_location is None:
        logger.info(f'{info.user_ip} No face found.')
        put_markdown('# 顔を検出できませんでした。')
        return

    index, similarity = sfs.get_best_match(nd_img, face_location)
    surrounded_img = Image.fromarray(sfs.surround_face(nd_img, face_location))
    logger.info(f'{info.user_ip} Best match: {sfs.model_data[index].name} ({similarity})')
    with use_scope('res', clear=True):
        put_markdown(f'# {sfs.model_data[index].name}との類似度は：{round(similarity*100,2)}%')
        put_row(
            content=[
                put_image(surrounded_img, width='200px'),
                put_column(
                    [
                        put_image(sfs.model_data[index].src_url, width='200px'),
                        put_html(sfs.model_data[index].comment),
                    ]
                )
            ]
        )


def run():
    log_path = os.path.abspath(f'{__file__}/../logs')
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging.basicConfig(
        level=logging.INFO,
        filename=f'{log_path}/log.log',
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )

    start_server(
        main,
        port=config.STAR_FACE_SIMILARITY_PORT,
        static_dir='./app/public',
        allowed_origins=['*'],
        # session_expire_seconds=600,
        # session_cleanup_interval=120,
        # max_payload_size='400M'
    )


if __name__ == '__main__':
    run()
