FROM --platform=linux/amd64 python:3.11-slim

# install updates
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy inference code
COPY requirements_training.txt /requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY reports/ reports/
#COPY data/ data/

# create directory for model storage
RUN mkdir -p models/
RUN mkdir -p data/processed/


WORKDIR /
# set dvc credentials
ENV DVC_SECRET=$DVC_SECRET

# install python packages and torch without gpu support
RUN apt-get remove swig
RUN apt-get install swig3.0
RUN ln -s /usr/bin/swig3.0 /usr/bin/swig
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -r /requirements.txt

# download data, run training and push the model
CMD ["python3", "src/models/train_model.py"]
