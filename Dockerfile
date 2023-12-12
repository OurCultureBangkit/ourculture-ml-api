FROM python:3.10-slim

WORKDIR /flask-app
COPY app.py /flask-app
COPY requirements.txt /flask-app
COPY labels.txt /flask-app
COPY model.h5 /flask-app

RUN pip3 install virtualenv
RUN python3 -m venv web-app
RUN . web-app/bin/activate
RUN python3 -m pip install -r requirements.txt

EXPOSE 5000
ENV PORT 5000

CMD exec gunicorn --bind :$PORT app:app --workers 1 --threads 8 --timeout 0