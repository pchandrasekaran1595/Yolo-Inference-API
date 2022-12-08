FROM python:3.8

WORKDIR /src/

COPY ./requirements.txt /src/

RUN pip install -r requirements.txt

COPY ./static /src/static

COPY ./main.py /src/

COPY ./test_main.py /src/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050", "--workers", "4"]