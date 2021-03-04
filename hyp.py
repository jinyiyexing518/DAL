# lr
lr0: 0.0001
warmup_lr: 0.00001   
warm_epoch:5


# setting
# HRSC2016 hrsc_dataset.py文件
# 分类的级别level1：1+1, level2：1+4, level4: 1+19
# 需要分类更多时，需要修改hrsc_dataset.py文件中的level
num_classes: 1

# training
epochs: 100
batch_size: 2
save_interval: 5
test_interval: 5
