file_name = "train.txt"
bert_path = "/home/chy/projects/mmaction2/work_dirs/bert_model"
allowed_keys = ["review_summary"]
batch_size = 4
epochs = 5


mlp_layer_num = 3
mlp_dims = (768, 256, 256, 3)
with_bn = True
act_type = 'relu'
last_w_bnact = False
last_w_softmax = True


sgd = dict(lr=0.01, momentum=0.9)

log = dict(iter=2)
