# set base image (host OS)
FROM python:3.8.6-slim

RUN  mkdir -p ~/project-api

WORKDIR ~/project-api/

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY . .

# command to run on container start
CMD [ "python", "./inference.py" ]
