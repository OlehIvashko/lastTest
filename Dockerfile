# Python 3.10
FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY app/ ./app
COPY lambda_handler.py .


CMD ["lambda_handler.lambda_handler"]
