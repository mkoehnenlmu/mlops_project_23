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

WORKDIR /
# set dvc credentials
ENV DVC_SECRET=$DVC_SECRET

# install python packages
RUN pip3 install --no-cache-dir -r /requirements.txt

# pull the data file from storage with dvc
COPY .dvc/ .dvc/
RUN dvc pull data/processed/data.csv

# run training
CMD ["python3", "src/models/train_model.py"]

# add trained model to dvc
RUN dvc add models/model.pth

# push trained model to storage
RUN dvc push models/model.pth
