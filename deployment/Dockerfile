FROM python:3.9-slim
COPY ./ ./app
WORKDIR ./app
RUN pip3 install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y


EXPOSE 8000
CMD ["uvicorn", "app:app","--host","0.0.0.0", "--port","8000" ]