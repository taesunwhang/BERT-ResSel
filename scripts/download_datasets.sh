#!/usr/bin/env bash

export file_name=ubuntu.zip
if [ -f $PWD/data/ubuntu_corpus_v1/ubuntu_train.pkl ]; then
    echo "ubuntu_train.pkl exists"
else
    echo "ubuntu_train.pkl does not exist"
    export file_id=1VKQaNNC5NR-6TwVPpxYAVZQp_nwK3d5u

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    unzip $file_name -d data/ubuntu_corpus_v1
    rm -r $file_name
fi

export file_name=ubuntu_post_training.txt
if [ -f $PWD/data/ubuntu_corpus_v1/$file_name ]; then
    echo "$file_name exists"
else
    echo "$file_name does not exist"
    export file_id=1mYS_PrnrKx4zDWOPTFhx_SeEwdumYXCK

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt
    mv $file_name data/ubuntu_corpus_v1/
fi

