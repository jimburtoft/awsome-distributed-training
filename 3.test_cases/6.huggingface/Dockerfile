FROM public.ecr.aws/w6p6i9i7/aws-efa-nccl-rdma:22.08-pt-py3
RUN HF_V=4.25.1 && git clone -b v${HF_V} https://github.com/huggingface/transformers.git /transformers && \
    pip install --upgrade pip && python -m pip install -e /transformers && python -m pip install datasets huggingface-hub sacrebleu evaluate
RUN python -m pip install deepspeed==0.7.7 accelerate==0.15.0
