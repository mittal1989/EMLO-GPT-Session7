FROM python:3.9-slim-buster

WORKDIR /workspace

COPY requirements.txt .
RUN pip install \
    -r requirements.txt \
    -f https://download.pytorch.org/whl/torch_stable.html \
    && rm -rf /root/.cache/pip

COPY setup.py .
RUN pip install -e .

COPY . .