#!/usr/bin/env bash
nvidia-smi
conda env create -f envrionment.yml
conda activate cocoenv
apt update
apt install default-jdk -y
python -W ignore train_search.py --batch_size 10 --layers 4 --data data/
