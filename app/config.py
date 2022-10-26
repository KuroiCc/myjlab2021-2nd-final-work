from pydantic import BaseSettings, FilePath


class Config(BaseSettings):

    MODEL_DATA_PATH: FilePath = "./app/data.csv"
    FACE_CASCADE_PATH: FilePath = "./app/haarcascade_frontalface_default.xml"
    LOADING_GIF_PATH: FilePath = "./app/public/loading.gif"
    STAR_FACE_SIMILARITY_PORT: int

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = True


config = Config()

if __name__ == "__main__":
    print(config.dict())
