FROM python:3.9

WORKDIR /code/

COPY ./requirements.txt /code/

RUN pip install -r requirements.txt

COPY ./static /code/static

COPY ./main.py /code/

COPY ./test_main.py /code/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6600", "--workers", "4"]