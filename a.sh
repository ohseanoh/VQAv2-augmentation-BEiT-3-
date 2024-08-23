#!/bin/bash

while true; do
    echo "processing..."
    git push -u origin main

    # 명령어가 성공했는지 확인
    if [ $? -eq 0 ]; then
        echo "success !!"
        break
    else
        echo "retrying..."
    fi

    # 짧은 대기 시간(예: 5초)을 두고 다시 시도할 수 있습니다.
    sleep 5
done

