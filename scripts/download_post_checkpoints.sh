#!/usr/bin/env bash

export file_name=bert-post-uncased-pytorch_model.pth
if [ -f $PWD/resources/bert-post-uncased/$file_name ]; then
    echo "$file_name exists"
else
    echo "$file_name does not exist"
    export file_id=1jt0RhVT9y2d4AITn84kSOk06hjIv1y49

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    mv $file_name resources/bert-post-uncased/
fi
