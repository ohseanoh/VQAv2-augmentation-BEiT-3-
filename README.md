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

```bash
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_large_patch16_480 \
        --input_size 480 \
        --task vqav2 \
        --batch_size 16 \
        --layer_decay 1.0 \
        --lr 2e-5 \
        --update_freq 1 \
        --randaug \
        --epochs 10 \
        --warmup_epochs 1 \
        --drop_path 0.15 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_large_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.01 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --task_head_lr_weight 20 \
        --opt_betas 0.9 0.98 \
        --enable_deepspeed \
        --checkpoint_activations
```
