ARG BASE_IMAGE

FROM ${BASE_IMAGE}

LABEL description="Latest PyTorch DALI container"

ENV PYTHONUNBUFFERED=TRUE

ENV PYTHONDONTWRITEBYTECODE=TRUE

COPY requirements.txt /workspace/

COPY src /workspace/
    
RUN pip3 install -r /workspace/requirements.txt 

WORKDIR /workspace

