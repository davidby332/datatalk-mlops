FROM python:3.10

RUN apt-get install wget

COPY requirements.txt requirements.txt  

RUN pip install -r requirements.txt

WORKDIR /usr/src/app

COPY . .

CMD ["tail", "-f", "/dev/null"]