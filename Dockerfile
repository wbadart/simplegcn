FROM python:3.7-slim

COPY . /usr/src
WORKDIR /usr/src

RUN pip install -r requirements.txt pytest && pip install -e .
CMD python -m pytest -v tests.py
