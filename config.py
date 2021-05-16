train_file = "data/train_split.txt"
val_file = "data/val.txt"
test_file = "data/test.txt"
bert_path = "bert_model"
ckpt_path = "ckpt/best_ckpt.pth"
allowed_keys = ["review_summary"]
batch_size = 64
num_workers = 5
epochs = 5
save_model = True
GPUS = 2


mlp_dims = (768, 256, 3)
mlp_layer_num = len(mlp_dims) - 1
with_bn = True
act_type = 'relu'
last_w_bnact = False
last_w_softmax = True


sgd = dict(lr=0.01, momentum=0.9)

log = dict(iter=2)
log_path = "log"

port = 29600
