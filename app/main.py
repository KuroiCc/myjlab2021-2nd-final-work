import os

from PIL import Image
from pywebio import start_server
from pywebio.input import file_upload
from pywebio.output import put_markdown, use_scope, put_image, put_html, put_column, put_row

from star_face_similarity import StarFaceSimilarity


def main():
    model_data_path = './app/data.csv'
    faceCascade_path = './app/haarcascade_frontalface_default.xml'
    sfs = StarFaceSimilarity(model_data_path, faceCascade_path)

    put_markdown('''# 有名人と顔の類似度
どの有名人と一番似ているかを比較します。
        ''')

    img = file_upload("Select a image:", accept="image/*", required=True)
    print(f'Accept pictures: {img["filename"]}')
    with use_scope('res', clear=True):
        with open('./app/public/loading.gif', 'rb') as f:
            loading = f.read()

        put_image(loading, width='300px')

    nd_img = sfs.get_rgb_ndarray_img_from_bytes(img['content'])
    face_location = sfs.find_one_face(nd_img)
    if face_location is None:
        put_markdown('# 顔を検出できませんでした。')
        return

    index, similarity = sfs.get_best_match(nd_img, face_location)
    surrounded_img = Image.fromarray(sfs.surround_face(nd_img, face_location))
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


if __name__ == '__main__':
    start_server(main, port=39001, static_dir='./app/public')
