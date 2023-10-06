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
# Install torch without gpu support
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -r /requirements.txt

# download model and run inference
CMD ["python3", "src/models/predict_model.py"]
