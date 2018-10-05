import os

os.system("python train.py --assets 0 -num_epoches 50 --dropout_keep_prob 0.6 --batch_size 32")
os.system("python train.py --assets 1 -num_epoches 50 --dropout_keep_prob 0.6 --batch_size 32")
os.system("python train.py --assets 2 -num_epoches 50 --dropout_keep_prob 0.6 --batch_size 32")
os.system("python train.py --assets 3 -num_epoches 50 --dropout_keep_prob 0.6 --batch_size 32")
