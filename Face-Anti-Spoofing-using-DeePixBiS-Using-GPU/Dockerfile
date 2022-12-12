FROM python:3.8-slim-buster

WORKDIR /app

COPY  requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "Test.py"]

#To build the Docker image for this model, you can run the following command:
#docker build -t pytorch-Model .

#Once the image has been built, you can run it as a container with the following command:
#docker run -it pytorch-Model

#This will create a container with your PyTorch anti-spoofing model, which you can then use to make predictions.
