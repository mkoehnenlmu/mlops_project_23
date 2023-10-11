FROM --platform=linux/amd64 python:3.11-slim

# install updates
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy inference code
COPY requirements_monitoring.txt /requirements.txt
COPY setup.py setup.py
COPY src/ src/

# create directory for model storage
RUN mkdir -p models/
RUN mkdir -p data/
RUN mkdir -p data/inference

# install python packages
WORKDIR /
# Install torch without gpu support
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -r /requirements.txt

# start app that runs inference
CMD ["uvicorn", "src.visualization.monitoring:app", "--host", "0.0.0.0", "--port", "81"]
