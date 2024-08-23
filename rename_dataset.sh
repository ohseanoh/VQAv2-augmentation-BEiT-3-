#!/bin/bash

# 원본 디렉토리와 대상 디렉토리 설정
src_dir="/data/Shared_Data/VQAv2/train2014"
dst_dir="/data/Shared_Data/VQAv2_aug/train2014"

# 대상 디렉토리가 존재하지 않으면 생성
mkdir -p "$dst_dir"

# 원본 디렉토리의 모든 jpg 파일을 순회
for file in "$src_dir"/*.jpg; do
    # 파일명에서 숫자 부분 추출 (정규 표현식을 사용하여 숫자만 추출)
    base_number=$(basename "$file" .jpg | grep -oP '(?<=COCO_train2014_)\d{12}$')
    
    # 숫자 부분이 잘 추출되었는지 확인
    if [[ -z "$base_number" ]]; then
        echo "Error: COCO_train2014_ 뒤에 숫자 부분을 확인할 수 없습니다. 파일: $file"
        continue
    fi
    
    # 600000을 더한 새로운 숫자 생성
    new_number=$(printf "%012d" $((10#$base_number + 600000)))
    
    # 새로운 파일명 생성
    new_file_name=$(basename "$file" | sed "s/${base_number}/${new_number}/")
    
    # 파일 복사
    cp "$file" "$dst_dir/$new_file_name"
done

