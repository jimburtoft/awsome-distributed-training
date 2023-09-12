HF_V=4.22.1 && git clone -b v${HF_V} https://github.com/huggingface/transformers.git /transformers && \
    pip install --upgrade pip && python -m pip install python-etcd==0.4.5 transformers==${HF_V} datasets huggingface-hub sacrebleu==2.1.0 evaluate==0.2.2 && \
    python -m torch.distributed.run /transformers/examples/pytorch/translation/run_translation.py \
        --model_name_or_path "t5-3b" \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap "T5Block" \
        --logging_steps 10 \
        --max_steps 100 \
        --do_train \
        --save_strategy "no" \
        --source_lang en \
        --target_lang ro \
        --source_prefix "translate English to Romanian: " \
        --dataset_name wmt16 \
        --dataset_config_name ro-en \
        --output_dir /tmp/tst-translation \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --overwrite_output_dir \
        --logging_dir ${TENSORBOARD_DIR} \
        --fp16 \
        --gradient_accumulation_steps 10 \
        --half_precision_backend cuda_amp