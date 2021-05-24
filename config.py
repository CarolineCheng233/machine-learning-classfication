train_file = "data/train_split.txt"
val_file = "data/val_split.txt"
test_file = "data/test.txt"
bert_path = "bert_model"
ckpt_dir = "ckpt"
ckpt_name = "best_ckpt.pth"
result_dir = "result"
result_name = "test_results.txt"
allowed_keys = ["review_text"]
batch_size = 64
num_workers = 10
epochs = 5
save_model = True

ratio = [5.75, 1, 5.75]
gamma = 2

bert_freeze = False
mlp_dims = (768, 3)
mlp_layer_num = len(mlp_dims) - 1
with_bn = True
act_type = 'relu'
last_w_bnact = False
last_w_softmax = False


sgd = dict(lr=2e-5, momentum=0.9, weight_decay=0.0005)

log = dict(iter=20)
log_path = "log"

