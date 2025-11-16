# draw_text_alignment.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from fairseq import checkpoint_utils, utils

# ====== 手动配置 ======
CKPT = "checkpoints/dy_transformer_8/checkpoint_best.pt"
DATA = "data-bin/en-de"
SPLIT = "test2016"

IDX = 10  # 想看的“数据集内部样本编号”，比如 0 / 32 / 694 等
USE_CUDA = torch.cuda.is_available()

# ====== 1. 加载模型和 task ======
print("==== DyTransformerModel module is loaded ====")
print("loading checkpoint...")
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [CKPT],
    arg_overrides={"data": DATA},
)
model = models[0]
model.eval()
if USE_CUDA:
    model.cuda()

src_dict = task.source_dictionary
tgt_dict = task.target_dictionary

# ====== 2. 加载数据集，先看看原始样本内容 ======
task.load_dataset(SPLIT)
dataset = task.dataset(SPLIT)

raw_item = dataset[IDX]
print(f"[INFO] raw sample idx={IDX}")
print("  src ids:", raw_item["source"])
print("  tgt ids:", raw_item["target"])

# ====== 3. 用 get_batch_iterator 找到包含这个 IDX 的 batch ======
itr = task.get_batch_iterator(
    dataset=dataset,
    max_tokens=4096,
    max_sentences=1,  # 每个 batch 只放 1 条，方便处理
    max_positions=utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    ),
    ignore_invalid_inputs=True,
    seed=1,
    num_workers=1,
).next_epoch_itr(shuffle=False)  # 不打乱

chosen_sample = None
for batch in itr:
    # batch["id"] 是这条样本在 dataset 里的索引
    b_id = batch["id"][0].item()
    if b_id == IDX:
        chosen_sample = batch
        break

if chosen_sample is None:
    raise RuntimeError(f"Cannot find sample with id={IDX} in iterator.")

sample = chosen_sample
if USE_CUDA:
    sample = utils.move_to_cuda(sample)

net_input = sample["net_input"]
src_tokens = net_input["src_tokens"]           # [1, Ts]
src_lengths = net_input["src_lengths"]
prev_output_tokens = net_input["prev_output_tokens"]  # [1, Tt]
img_features_list = net_input["img_features_list"]

print(f"[INFO] found batch with id={sample['id'][0].item()}")

# ====== 4. encoder 前向 ======
with torch.no_grad():
    encoder_out = model.encoder(
        src_tokens=src_tokens,
        src_lengths=src_lengths,
        img_features_list=img_features_list,
        return_all_hiddens=False,
    )

# ====== 5. decoder 前向，取 cross-attention 对齐矩阵 ======
decoder = model.decoder

with torch.no_grad():
    features, extra = decoder.extract_features(
        prev_output_tokens=prev_output_tokens,
        encoder_out=encoder_out,
        img_features_list=img_features_list,
        full_context_alignment=True,   # 不加自回归 mask，方便整体看对齐
        alignment_layer=None,          # 默认最后一层
        alignment_heads=None,          # 平均所有 head
    )

attn = extra["attn"][0]  # 一般已经是 [T_tgt, T_src]，有的版本可能是 [B, T_tgt, T_src]
print("raw attn shape:", attn.shape)

if attn.dim() == 3:
    # [B, T_tgt, T_src] -> 取 batch 中第一个样本
    attn = attn[0]

attn = attn.cpu().numpy()  # [T_tgt, T_src]

# ====== 6. 去掉 PAD，并获得 token 文本 ======
pad_src = src_dict.pad()
pad_tgt = tgt_dict.pad()

src_tokens_1 = src_tokens[0]
tgt_tokens_1 = prev_output_tokens[0]

src_valid_idx = [i for i, tok in enumerate(src_tokens_1) if tok != pad_src]
tgt_valid_idx = [i for i, tok in enumerate(tgt_tokens_1) if tok != pad_tgt]

attn = attn[np.ix_(tgt_valid_idx, src_valid_idx)]

def tensor_to_tokens(tensor, dictionary):
    words = []
    for tok in tensor:
        tok_id = int(tok)
        w = dictionary.string([tok_id]).strip()
        words.append(w)
    return words

src_words = tensor_to_tokens(src_tokens_1[src_valid_idx], src_dict)
tgt_words = tensor_to_tokens(tgt_tokens_1[tgt_valid_idx], tgt_dict)

print("src sentence:", " ".join(src_words))
print("tgt sentence:", " ".join(tgt_words))
print("attn final shape:", attn.shape)

# ====== 7. 画热图 ======
plt.figure(figsize=(10, 5))
plt.imshow(attn, aspect="auto", cmap="Blues", origin="upper")

plt.xticks(
    ticks=range(len(src_words)),
    labels=src_words,
    rotation=45,
    ha="right",
    fontsize=8,
)
plt.yticks(
    ticks=range(len(tgt_words)),
    labels=tgt_words,
    fontsize=8,
)

plt.xlabel("Source tokens")
plt.ylabel("Target tokens (decoding step t)")
plt.colorbar(label="Cross-attention weight")

plt.tight_layout()
os.makedirs("align_vis", exist_ok=True)
out_path = os.path.join("align_vis", f"align_sample_{IDX}.png")
plt.savefig(out_path, dpi=300)
plt.close()
print("saved:", out_path)
