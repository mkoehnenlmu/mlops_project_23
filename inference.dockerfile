FROM --platform=linux/amd64 python:3.11-slim

# install updates
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy inference code
COPY requirements_docker.txt /requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

# create directory for model storage
RUN mkdir -p models/

# install python packages
WORKDIR /
RUN pip3 install --no-cache-dir -r /requirements.txt

# pull the model file from storage with dvc
COPY .dvc/ .dvc/
# TODO: How does dvc know the model md5 hash
# if we change the model file?
RUN dvc pull models/model.pth

# run inference
CMD ["python3", "src/models/predict_model.py"]
