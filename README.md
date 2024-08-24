# VQAv2-augmentation-BEiT-3-

# Data augmentation for visual question answering

implementing data augmentation for better VQA.

## 목차

- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [기여 방법](#기여-방법)
- [라이선스](#라이선스)
- [연락처](#연락처)

## 환경 설정

1. set up:
    ```bash
    git clone https://github.com/microsoft/unilm.git
    cd unilm/beit3
    conda create -n beit pythone=3.8 -y
    pip install -r requirements.txt
    pip install protobuf==3.20.*
    utils.py의 24번째 줄 torch._six -> torch
    ```
2. custom dataset & models:
    ```bash
    datasets.py의 701번째 줄 task2dataset에 "vqav2_aug": VQAv2Dataset, "vqav2_flip": VQAv2Dataset, 추가
    engine_for_finetuning.py의 442번째 줄 get_handler에 elif args.task == "vqav2" or args.task == "vqav2_aug" or args.task == "vqav2_flip"
    modeling_finetune.py의 @register_model에 answer2label의 class수에 맞는 num_classes 할당한 모델들 추가
    run_beit3_finetuning.py의 38번째 줄 parser.add_argument('--task'에 'vqav2_aug', 'vqav2_flip', 추가
    run_beit3_finetuning.py의 367번째 줄 args.task == "vqav2_aug", args.task == "vqav2_flip" 추가
    ```
3. data augmentation:
    ```bash
    이미지 augmentation: python augment.py > output.log 2> error.log
    텍스트 augmentation: json_arranging.ipynb
    index_file 생성: python generate_index_files.py > output.log 2> error.log
    ```

## Command

inference:
    '''bash
    TIMESTAMP=$(date +'%Y%m%d_%H%M')
    LOGFILE="/home/seanoh/unilm/beit3/logs/evaluation_${TIMESTAMP}.log"
    OUTPUT_DIR="/home/seanoh/unilm/beit3/predictions"
    export OMP_NUM_THREADS=16
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 /home/seanoh/unilm/beit3/run_beit3_finetuning.py \
        --model beit3_base_patch16_480 \
        --input_size 480 \
        --task vqav2 \
        --batch_size 16 \
        --sentencepiece_model /home/seanoh/unilm/beit3/models/beit3.spm \
        --finetune /home/seanoh/unilm/beit3/finetuned/checkpoint_best_flip.pth \
        --data_path /data/Shared_Data/VQAv2_flip \
        --output_dir "${OUTPUT_DIR}" \
        --eval \
        --dist_eval 2>&1 | tee "$LOGFILE"
    mv "${OUTPUT_DIR}/submit_vqav2_test.json" "${OUTPUT_DIR}/submit_vqav2_test_flip2flip_${TIMESTAMP}.json"
    '''
    
train:
    ```bash
    TIMESTAMP=$(date +'%Y%m%d_%H%M')
    LOGFILE_DIR="/home/seanoh/unilm/beit3/logs/finetuning_${TIMESTAMP}"
    LOGFILE="${LOGFILE_DIR}.log"
    export OMP_NUM_THREADS=16
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 /home/seanoh/unilm/beit3/run_beit3_finetuning.py \
        --model beit3_base_patch16_480_vqav2 \
        --input_size 480 \
        --task vqav2 \
        --batch_size 8 \
        --layer_decay 1.0 \
        --lr 3e-5 \
        --update_freq 1 \
        --epochs 10 \
        --warmup_epochs 1 \
        --drop_path 0.1 \
        --sentencepiece_model /home/seanoh/unilm/beit3/models/beit3.spm \
        --finetune /home/seanoh/unilm/beit3/models/beit3_base_patch16_224.pth \
        --data_path /data/Shared_Data/VQAv2 \
        --output_dir "/home/seanoh/unilm/beit3/finetuned" \
        --log_dir "$LOGFILE_DIR" \
        --weight_decay 0.01 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --task_head_lr_weight 20 \
        --opt_betas 0.9 0.98 \
    2>&1 | tee "$LOGFILE"
    '''
