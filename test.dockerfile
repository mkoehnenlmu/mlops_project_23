FROM --platform=linux/amd64 python:3.11-slim

# copy inference code
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

# install python packages
COPY requirements_tests.txt requirements.txt

WORKDIR /

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["pytest", "-v"]