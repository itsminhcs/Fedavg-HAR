## Dataset
The Human Activity Recognition (HAR) dataset used in this project can be downloaded from the following link:
🔗 [Download HAR Dataset](https://drive.google.com/drive/u/3/folders/1X6L4y2t2TwjkTAUfEE0ANBusTzNV27LA)
## Requirements
pip install -r requirements.txt
## Train with iid
python federated_main.py --model=cnn --dataset=har --epochs=300 --frac 0.1 --local_ep=20 --lr=0.0003 --optimizer='adam' --num_users=100 --iid=1 --gpu='0' --data_dir='data/' --output_dir='output_dir/' --local_bs=128
## Train with non-iid
python federated_main.py --model=cnn --dataset=har --epochs=300 --frac 0.1 --local_ep=20 --lr=0.0003 --optimizer='adam' --num_users=100 --iid=0 --gpu='0' --data_dir='data/' --output_dir='output_dir/' --local_bs=128
