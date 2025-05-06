import json
import numpy as np
import tensorflow as tf
from recommenders.models.sasrec.model import SASREC


# Load config
with open("data/ckpt/sasrec_config.json") as f:
    config = json.load(f)

# Recreate model
model = SASREC(
    item_num=config["item_num"],
    seq_max_len=config["seq_max_len"],
    num_blocks=config["num_blocks"],
    embedding_dim=config["embedding_dim"],
    attention_dim=config["attention_dim"],
    attention_num_heads=config["attention_num_heads"],
    dropout_rate=config["dropout_rate"],
    conv_dims=config["conv_dims"],
    l2_reg=config["l2_reg"],
    num_neg_test=config["num_neg_test"]
)

# Restore weights
ckpt = tf.train.Checkpoint(model=model)
ckpt.restore("data/ckpt/sasrec.ckpt").expect_partial()

# 1) Restore your trained model
ckpt = tf.train.Checkpoint(model=model)
ckpt.restore(str(CKPT_DIR / "sasrec.ckpt")).expectP_partial()
model.eval()  # disable dropout / training hooks

# 2) Helper to get top-K for any user
def recommend_for_user(user_id, K=10):
    # ---- build the user sequence ----
    # assume `dataset.user_train` is a dict: user → [item_id,…]
    hist = dataset.user_train[user_id]            # your training history list
    seq = np.zeros((1, seq_max_len), dtype=np.int32)
    seq[0, -len(hist):] = hist                    # right-align the history

    # ---- build the candidate set ----
    # score *all* items (1…itemnum)
    all_items = np.arange(1, dataset.itemnum + 1, dtype=np.int32)
    candidates = all_items[np.newaxis, :]         # shape (1, itemnum)

    # ---- predict and mask seen ----
    inputs = {"input_seq": seq, "candidate": candidates}
    scores = model.predict(inputs).numpy()[0]      # shape: (itemnum + 1)
    scores[hist] = -np.inf                        # don’t recommend what’s seen

    # ---- pick top-K ----
    topk = np.argsort(scores)[::-1][:K] + 1       # +1 if your IDs are 1-based
    return topk

# 3) Try it out
user_id = 123
print(f"Top 10 for user {user_id}:", recommend_for_user(user_id, K=10))