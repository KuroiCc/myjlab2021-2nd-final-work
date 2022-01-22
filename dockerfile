FROM animcogn/face_recognition:cpu-latest

RUN apt-get -y update && \
    apt-get install -y git libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    git clone https://github.com/KuroiCc/myjlab2021-2nd-final-work.git app && \
    cd app && \
    /opt/venv/bin/python3 -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app
ENV APP_PORT=${APP_PORT:-80}

WORKDIR /app

EXPOSE 80

ENTRYPOINT [ "python", "app/main.py" ]