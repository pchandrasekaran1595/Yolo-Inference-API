FROM python:3.8

WORKDIR /code/

COPY ./requirements.txt /code/

RUN pip install -r requirements.txt

COPY ./static /code/static

COPY ./main.py /code/

COPY ./test_main.py /code/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050", "--workers", "4"]