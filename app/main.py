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
    res = sfs.get_best_match_from_bytes(img['content'])
    if res is None:
        put_markdown('# 顔を検出できませんでした。')
    else:
        name, similarity, src_url, comment = res
        with use_scope('res', clear=True):
            put_markdown(f'# {name}との類似度は：{round(similarity*100,2)}%')
            put_row(
                content=[
                    put_image(img['content'], width='200px'),
                    put_column([
                        put_image(src_url, width='200px'),
                        put_html(comment),
                    ])
                ]
            )


if __name__ == '__main__':
    start_server(main, port=39001, debug=True)
