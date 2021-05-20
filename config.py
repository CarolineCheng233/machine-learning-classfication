train_file = "data/train_resample.txt"
val_file = "data/val.txt"
test_file = "data/test_with_label.txt"
bert_path = "bert_model"
ckpt_dir = "ckpt"
ckpt_name = "best_ckpt.pth"
result_dir = "result"
result_name = "test_results.txt"
allowed_keys = ["review_summary"]
batch_size = 64
num_workers = 4
epochs = 20
save_model = True

ratio = [5.75, 1, 5.75]
gamma = 2

bert_freeze = True
mlp_dims = (768, 3)
mlp_layer_num = len(mlp_dims) - 1
with_bn = True
act_type = 'relu'
last_w_bnact = False
last_w_softmax = True


sgd = dict(lr=0.01, momentum=0.9)

log = dict(iter=20)
log_path = "log"

