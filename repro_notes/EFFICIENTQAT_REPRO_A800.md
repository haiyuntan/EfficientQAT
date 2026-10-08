# EfficientQAT 单卡 A800 复现手册（R0–R5）

> 项目目标：先在 **Llama-2-7B** 上复现 EfficientQAT 的两阶段流程（Block-AP → E2E-QP），确认实现、数据、量化与评测链路正确；然后必须完成 Alpaca Instruction Tuning 迁移实验（R5A），最终迁移到 REAL-Prover prover 模型。
>
> 目标硬件：**1 × NVIDIA A800 80GB**
>
> 上游仓库：https://github.com/OpenGVLab/EfficientQAT
> 工作 fork：https://github.com/haiyuntan/EfficientQAT
> 论文：https://arxiv.org/abs/2407.11062

---

# 1. 第一阶段复现的目标

第一阶段不是重跑论文所有表格，而是验证以下 4 件事：

1. 官方环境和 evaluation pipeline 可复现；
2. Block-AP 可以在单卡 A800 上正确完成；
3. E2E-QP 可以加载 Block-AP checkpoint，并且只训练 quantization step size；
4. 我们训练出的 Llama-2-7B W2g64 性能接近官方结果。

## 1.1 核心实验矩阵

| ID | 模型                         | 量化   | 阶段                             | 目的                      |
| -- | ---------------------------- | ------ | -------------------------------- | ------------------------- |
| R0 | Llama-2-7B                   | FP16   | Eval only                        | FP baseline               |
| R1 | 官方 EfficientQAT checkpoint | W2g64  | Eval only                        | 验证评测环境              |
| R2 | 我们训练                     | W4g128 | Block-AP + E2E-QP                | W4 sanity check         |
| R3 | 我们训练                     | W2g64  | Block-AP only                    | 验证 Stage 1              |
| R4 | 我们训练                     | W2g64  | Block-AP + E2E-QP                | **核心复现结果**    |
| R5 | Llama-2-7B（扩展实验）       | W2g64  | 复用 R3 Block-AP + Alpaca E2E-QP | Instruction Tuning / MMLU |

**本轮必须完成 R0–R5；R5A 与 R4 同为高优先级验收项。R2 也包含在全流水线中。R5B（Llama-1 严格复现）保持后续独立工作。**

本轮包含 R2（W4g128），用于交叉验证量化链路。

---

## 1.2 官方参考结果

官方 README 对 Llama-2-7B 给出的参考值：

| Model      |           Quant |  WikiText2 PPL | 5-task Avg. Accuracy |
| ---------- | --------------: | -------------: | -------------------: |
| Llama-2-7B |            FP16 |           5.47 |                64.86 |
| Llama-2-7B |          W4g128 |           5.53 |                64.27 |
| Llama-2-7B |          W3g128 |           5.81 |                64.02 |
| Llama-2-7B | **W2g64** | **6.86** |      **60.14** |

5 个 zero-shot task：

- PIQA
- ARC-Easy
- ARC-Challenge
- HellaSwag
- WinoGrande

lm-eval 版本：**0.4.2**。

> 注意：不要求第一次运行逐位完全一致。首先判断 pipeline 是否正确、趋势是否一致；若偏差明显，再逐项定位随机种子、数据缓存、模型版本和 evaluation environment。

---

# 2. 单卡 A800 是否够用

够。

论文在单张 A100-80GB 上报告 Llama-2-7B：

- Block-AP：约 **3.3 h / 8.5 GB**
- E2E-QP：约 **1.5 h / 5.6–7.0 GB**
- 总计：约 **4.8 h**

A800-80GB 从显存容量上完全足够。

实际服务器速度会受到：

- CPU
- NVMe / 网络盘
- Hugging Face dataset cache
- DataLoader / preprocessing
- 当前代码版本

影响，因此**不要把 4.8h 当作严格运行时间保证**。

---

# 3. 推荐目录结构

建议服务器上建立：

```text
/workspace/qat/
├── EfficientQAT/              # git repo
├── models/
│   └── Llama-2-7b-hf/
├── hf_cache/
├── outputs/
│   ├── official_w2g64/
│   ├── block_ap/
│   └── e2e_qp/
├── logs/
└── env/
```

根据实际磁盘位置修改：

```bash
export QAT_ROOT=/workspace/qat
mkdir -p \
  $QAT_ROOT/models \
  $QAT_ROOT/hf_cache \
  $QAT_ROOT/outputs \
  $QAT_ROOT/logs \
  $QAT_ROOT/env
```

将 Hugging Face cache 放到本地高速盘：

```bash
export HF_HOME=$QAT_ROOT/hf_cache
export HF_DATASETS_CACHE=$QAT_ROOT/hf_cache/datasets
export TRANSFORMERS_CACHE=$QAT_ROOT/hf_cache/transformers
```

如果服务器是临时租用，建议写进自己的 shell 初始化文件，或者每次启动 session 后重新 export。

---

# 4. Git 仓库与工作流

## 4.1 从官方上游拉取

```bash
cd $QAT_ROOT

git clone https://github.com/OpenGVLab/EfficientQAT.git
cd EfficientQAT
```

记录当前上游 commit：

```bash
git rev-parse HEAD
git log -1 --oneline
```

把这两个结果写进实验日志。

---

## 4.2 配置 upstream + 自己的 fork

将官方仓库命名为 `upstream`：

```bash
git remote rename origin upstream
```

将自己的 fork 作为 `origin`。

推荐 SSH：

```bash
git remote add origin git@github.com:haiyuntan/EfficientQAT.git
```

检查：

```bash
git remote -v
```

预期类似：

```text
origin    git@github.com:haiyuntan/EfficientQAT.git
upstream  https://github.com/OpenGVLab/EfficientQAT.git
```

---

## 4.3 创建复现分支

```bash
git switch -c reproduce/llama2-7b-w2g64
```

第一次 push：

```bash
git push -u origin reproduce/llama2-7b-w2g64
```

后续所有代码修改都在这个分支进行。

### 建议 commit 粒度

```text
repro: record environment and baseline
repro: add explicit w2g64 block-ap script
repro: add explicit w2g64 e2e-qp script
fix: ...
exp: ...
```

每完成一个可运行 milestone：

```bash
git status
git add <需要提交的代码/配置/小型日志>
git commit -m "..."
git push
```

---

# 5. 不要上传到 GitHub 的内容

模型权重、cache 和大型 checkpoint 不应该进入 Git。

建议检查或补充 `.gitignore`：

```gitignore
# models / data
models/
hf_cache/
cache/

# experiment artifacts
output/
outputs/
logs/*.log

# checkpoints
checkpoint-*/
*.bin
*.safetensors
*.pt
*.pth

# Python
__pycache__/
*.pyc
```

**应该提交：**

- 修改后的 Python 源码
- shell scripts
- config
- 环境版本记录
- 小型 metrics JSON
- 实验摘要 Markdown
- bug fix

**不应该提交：**

- Llama 权重
- RedPajama cache
- Block-AP quantized checkpoint
- E2E checkpoint

---

# 6. 检查服务器硬件

```bash
nvidia-smi
```

记录：

```bash
nvidia-smi > $QAT_ROOT/env/nvidia-smi.txt
```

确认：

- GPU = A800
- VRAM ≈ 80GB
- Driver 正常
- CUDA 可见

检查 CPU / 内存 / 磁盘：

```bash
lscpu | tee $QAT_ROOT/env/lscpu.txt
free -h | tee $QAT_ROOT/env/memory.txt
df -h | tee $QAT_ROOT/env/disk.txt
```

---

# 7. 创建官方环境

官方仓库当前 `requirements.txt` 固定：

```text
accelerate==0.28.0
bitsandbytes==0.41.0
datasets==2.18.0
lm_eval==0.4.2
numpy==1.23.4
torch==2.2.2
tqdm==4.64.1
transformers==4.40.1
triton==2.2.0
termcolor
sentencepiece
protobuf
```

## 7.1 Conda

```bash
conda create -n efficientqat python=3.11 -y
conda activate efficientqat
```

建议更新基础工具：

```bash
python -m pip install --upgrade pip setuptools wheel
```

安装官方依赖：

```bash
cd $QAT_ROOT/EfficientQAT
pip install -r requirements.txt
```

---

## 7.2 环境 sanity check

```bash
python - <<'PY'
import torch
import transformers
import datasets
import lm_eval
import accelerate

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("bf16:", torch.cuda.is_bf16_supported())
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
PY
```

必须至少确认：

```text
cuda available: True
gpu: NVIDIA A800 ...
transformers: 4.40.1
datasets: 2.18.0
accelerate: 0.28.0
```

记录完整环境：

```bash
pip freeze > $QAT_ROOT/env/pip-freeze.txt
python --version > $QAT_ROOT/env/python-version.txt
git rev-parse HEAD > $QAT_ROOT/env/efficientqat-commit.txt
```

---

# 8. 获取 Llama-2-7B

我们第一阶段按照论文，使用 **Llama-2-7B**。

确保当前账号已经具备 `meta-llama/Llama-2-7b-hf` 的访问权限，然后下载：

```bash
export MODEL_DIR=$QAT_ROOT/models/Llama-2-7b-hf

huggingface-cli download meta-llama/Llama-2-7b-hf \
  --local-dir $MODEL_DIR
```

如果当前环境使用新版 CLI，也可以使用等价的：

```bash
hf download meta-llama/Llama-2-7b-hf \
  --local-dir $MODEL_DIR
```

确认：

```bash
ls -lh $MODEL_DIR
```

---

# 9. 数据：RedPajama

Block-AP 官方代码使用：

```text
togethercomputer/RedPajama-Data-1T-Sample
```

代码会通过 `datasets.load_dataset()` 自动获取。

Block-AP 默认：

```text
train_size       = 4096
val_size         = 64
training_seqlen  = 2048
seed             = 2
batch_size       = 2
epochs           = 2
```

第一次运行会下载和 tokenize 数据，后续会在 repo 的 `./cache` 下复用 dataloader cache。

建议确保：

```bash
mkdir -p $QAT_ROOT/EfficientQAT/cache
```

---

# 10. R0：先测 FP16 baseline

在训练前先确认原始 Llama-2-7B 评测能正常运行。

可以使用 repo 自己的 evaluation 路径，也可以单独记录 lm-eval baseline。

核心参考目标：

```text
WikiText2 PPL ≈ 5.47
5-task Avg Accuracy ≈ 64.86
```

如果 FP baseline 与官方差很多，**先不要开始 QAT**。

优先排查：

- Llama checkpoint 是否一致
- tokenizer
- lm_eval==0.4.2
- context length
- 评测任务名
- 是否 zero-shot
- dependency versions

---

# 11. R1：评估官方 W2g64 checkpoint

这是非常重要的 sanity check：

> 在自己开始训练前，先验证“官方模型 + 我们的环境”能不能复现官方结果。

安装/确认 Hugging Face CLI：

```bash
pip install huggingface_hub
```

下载：

```bash
mkdir -p $QAT_ROOT/outputs/official_w2g64

huggingface-cli download \
  ChenMnZ/Llama-2-7b-EfficientQAT-w2g64 \
  --local-dir $QAT_ROOT/outputs/official_w2g64
```

评估：

```bash
cd $QAT_ROOT/EfficientQAT

CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
  --resume_quant $QAT_ROOT/outputs/official_w2g64 \
  --net Llama-2 \
  --wbits 2 \
  --group_size 64 \
  --output_dir $QAT_ROOT/logs/official_w2g64_eval \
  --eval_ppl \
  --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
```

目标：

```text
WikiText2 PPL ≈ 6.86
5-task Avg Accuracy ≈ 60.14
```

### Stop condition

如果这个 checkpoint 的结果明显对不上官方数字：

**停止训练，先修 evaluation environment。**

否则我们无法判断后续“训练复现失败”还是“评测本身错误”。

---

# 12. R3：Block-AP — W2g64

这是 EfficientQAT Stage 1。

## 12.1 Block-AP 训练什么

当前 Transformer block 内：

```text
W : trainable
s : trainable
z : trainable
```

每次只训练一个 block。

优化目标是 FP block 与 quantized block 的 reconstruction。

---

## 12.2 论文 / 官方参数

W2g64：

```text
wbits              = 2
group_size         = 64

train_size         = 4096
val_size           = 64
training_seqlen    = 2048
batch_size         = 2
epochs             = 2

quant_lr (s,z)     = 1e-4
weight_lr (W)      = 2e-5
```

---

## 12.3 推荐：不用隐式默认值，全部显式传参

```bash
cd $QAT_ROOT/EfficientQAT

mkdir -p \
  $QAT_ROOT/outputs/block_ap/Llama-2-7b-w2g64 \
  $QAT_ROOT/logs/block_ap/Llama-2-7b-w2g64

CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
  --model $MODEL_DIR \
  --net Llama-2 \
  --wbits 2 \
  --group_size 64 \
  --calib_dataset redpajama \
  --train_size 4096 \
  --val_size 64 \
  --training_seqlen 2048 \
  --batch_size 2 \
  --epochs 2 \
  --quant_lr 1e-4 \
  --weight_lr 2e-5 \
  --real_quant \
  --output_dir $QAT_ROOT/logs/block_ap/Llama-2-7b-w2g64 \
  --save_quant_dir $QAT_ROOT/outputs/block_ap/Llama-2-7b-w2g64 \
  --eval_ppl \
  --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
  2>&1 | tee $QAT_ROOT/logs/block_ap_w2g64_console.log
```

---

## 12.4 训练中监控

另开一个 tmux pane：

```bash
watch -n 1 nvidia-smi
```

或者：

```bash
nvidia-smi dmon -s pucvmet
```

需要记录：

- GPU memory
- utilization
- 每个 block 花费时间
- train reconstruction loss
- validation reconstruction loss
- 是否 NaN
- 最终 PPL
- 最终 zero-shot accuracy

### 建议使用 tmux

```bash
tmux new -s efficientqat
```

避免 SSH 断连导致训练中止。

---

## 12.5 Block-AP 成功标准

至少满足：

- 所有 Transformer blocks 都训练完成；
- 没有 NaN / inf；
- quantized checkpoint 可以重新加载；
- PPL 没有异常爆炸；
- zero-shot accuracy 明显优于简单崩坏状态；
- 保存 Block-AP-only 结果，**不要被 E2E-QP 覆盖**。

Block-AP-only checkpoint 是之后验证 E2E-QP 是否真的有贡献的重要 baseline。

---

# 13. R4：E2E-QP — W2g64

这是 EfficientQAT Stage 2。

输入：

```text
Block-AP 得到的 Wq*
```

整个模型重新端到端运行，但默认：

```text
Wq : frozen
z  : frozen
s  : trainable
```

官方当前实现中会将 `QuantLinear.scales.requires_grad = True`，optimizer 只收集 scale 参数。

---

## 13.1 E2E-QP 参数

论文 / 官方 W2 配置：

```text
dataset                     = redpajama
max_train_samples           = 4096
pt_context_len              = 4096

per_device_train_batch_size = 4
gradient_accumulation_steps = 8
effective batch size        = 32

num_train_epochs            = 1
learning_rate(s)            = 2e-5

eval_dataset_size           = 64
bf16                        = True
```

---

## 13.2 运行

```bash
cd $QAT_ROOT/EfficientQAT

mkdir -p \
  $QAT_ROOT/outputs/e2e_qp/Llama-2-7b-w2g64 \
  $QAT_ROOT/logs/e2e_qp

CUDA_VISIBLE_DEVICES=0 python main_e2e_qp.py \
  --quant_model_path $QAT_ROOT/outputs/block_ap/Llama-2-7b-w2g64 \
  --model_family Llama-2 \
  --wbits 2 \
  --group_size 64 \
  --learning_rate 2e-5 \
  --dataset redpajama \
  --dataset_format pt \
  --output_dir $QAT_ROOT/outputs/e2e_qp/Llama-2-7b-w2g64 \
  --do_train True \
  --pt_context_len 4096 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --logging_steps 1 \
  --save_strategy epoch \
  --training_strategy epochs \
  --evaluation_strategy steps \
  --eval_steps 64 \
  --max_train_samples 4096 \
  --num_train_epochs 1 \
  --eval_dataset_size 64 \
  --bf16 \
  --data_seed 42 \
  --max_grad_norm 0.3 \
  --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
  --preprocessing_num_workers 32 \
  --do_ppl_eval \
  2>&1 | tee $QAT_ROOT/logs/e2e_qp_w2g64_console.log
```

> 如果 CPU 核数不足，`--preprocessing_num_workers 32` 可以降低；不要为了复现盲目开 32 个 worker 导致 CPU/RAM 压力。

---

# 14. E2E-QP 后必须检查 trainable parameter

运行日志中确认：

```text
只有 quantization scale 是 trainable
```

对应代码应满足：

```python
if isinstance(module, QuantLinear) and not 'head' in name:
    module.scales.requires_grad = True
```

且 optimizer 收集：

```python
[p for n, p in model.named_parameters() if 'scale' in n]
```

如果发现 full weight 也在训练：

**立刻停止。**

那就已经不是论文的 E2E-QP 了。

---

# 15. 结果对比

最终至少形成：

| Run                   | WikiText2 PPL | C4 PPL | PIQA | ARC-E | ARC-C | HellaSwag | WinoGrande | Avg |
| --------------------- | ------------: | -----: | ---: | ----: | ----: | --------: | ---------: | --: |
| FP16                  |               |        |      |       |       |           |            |     |
| Official W2g64        |               |        |      |       |       |           |            |     |
| Our Block-AP          |               |        |      |       |       |           |            |     |
| Our Block-AP + E2E-QP |               |        |      |       |       |           |            |     |

核心比较：

```text
FP16
  ↓
Block-AP
  ↓
Block-AP + E2E-QP
```

我们要回答：

1. W2 量化造成多少性能损失？
2. Block-AP 能恢复多少？
3. E2E-QP 是否进一步恢复？
4. 最终结果与官方 W2g64 的差距有多大？

---

# 16. 需要保存的实验 metadata

每次正式 run 都保存：

```text
date/time
hostname
GPU model
GPU driver
CUDA runtime
Python version
pip freeze
git commit hash
model path / exact HF model
dataset
seed
wbits
group_size
train_size
val_size
sequence length
batch size
gradient accumulation
epochs
learning rates
wall-clock time
peak GPU memory
final metrics
```

推荐建立：

```text
repro_notes/
├── environment.md
├── R0_fp16.md
├── R1_official_w2g64.md
├── R3_block_ap_w2g64.md
├── R4_e2e_qp_w2g64.md
└── results.md
```

这些 Markdown 和小型 metrics 文件可以提交 GitHub。

---

# 17. 建议增加一个结果记录模板

`repro_notes/RUN_TEMPLATE.md`：

```markdown
# Run ID

## Git

- Commit:
- Branch:

## Hardware

- GPU:
- Driver:
- CUDA:

## Model

- Model:
- Path:

## Quantization

- wbits:
- group_size:

## Data

- dataset:
- train samples:
- val samples:
- context length:

## Optimization

- batch:
- grad accumulation:
- epochs:
- W LR:
- quant LR:
- scale LR:

## Runtime

- start:
- end:
- wall time:
- peak GPU memory:

## Result

- WikiText2 PPL:
- C4 PPL:
- PIQA:
- ARC-E:
- ARC-C:
- HellaSwag:
- WinoGrande:
- Avg:

## Notes / problems

-
```

---

# 18. 建议先创建显式的复现脚本

官方 shell script 会依赖部分 Python 默认值。

为了后续迁移 REAL-Prover，推荐自己增加：

```text
repro_scripts/
├── eval_official_w2g64.sh
├── block_ap_llama2_7b_w2g64.sh
└── e2e_qp_llama2_7b_w2g64.sh
```

脚本里：

- 所有关键参数全部显式填写；
- model path 用环境变量；
- output/log path 用环境变量；
- 提交 GitHub。

例如顶部：

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${QAT_ROOT:?Please set QAT_ROOT}"
: "${MODEL_DIR:?Please set MODEL_DIR}"
```

这样以后迁移 REAL-Prover 时可以非常明确地 diff：

```text
Llama-2 baseline
vs.
REAL-Prover experiment
```

---

# 19. Git 同步规范

每次修改代码后：

```bash
git diff
git status
```

确认没有：

- model weight
- cache
- 大型 checkpoint

然后：

```bash
git add <files>
git commit -m "repro: <description>"
git push
```

为了让我后续通过网页准确读取修改，沟通时给出：

```text
branch:
commit SHA:
run ID:
```

例如：

```text
branch: reproduce/llama2-7b-w2g64
commit: abc1234
run: R3
```

这样可以精确对应代码和实验结果。

---

# 20. 从 upstream 同步最新修改

不要直接在 `main` 上做实验。

需要同步官方更新时：

```bash
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
```

然后再回实验分支：

```bash
git switch reproduce/llama2-7b-w2g64
git rebase main
```

如果复现过程中为了忠实论文需要固定某个 commit，则**不要随意同步最新 main**；优先保持实验代码版本冻结。

---

# 21. 可选：W4g128 sanity check

如果 W2 debugging 困难，可以先做 W4g128。

论文参数原则：

```text
wbits      = 4
group_size = 128

quant_lr   = 1e-4
weight_lr  = 1e-5

E2E scale LR = 1e-5
```

官方参考：

```text
WikiText2 PPL = 5.53
Avg Accuracy  = 64.27
```

W4 更容易判断量化实现有没有基础错误。

---

# 22. 第一阶段暂时不做的实验

为了控制 GPU 成本，第一轮不要立即复现：

- 所有 7B / 13B / 70B 模型
- W3 全套
- sample scaling ablation
- Table 5 / Table 6 所有 trainable-parameter ablation
- GPTQ / BitBLAS format transfer
- REAL-Prover
- task-aware QAT

先把：

```text
Llama-2-7B W2g64
Block-AP → E2E-QP → evaluation
```

跑通。

---

# 23. 第一阶段验收标准（Gate）

只有以下条件都满足，才进入 REAL-Prover：

## Gate A：环境

- [ ] A800 CUDA / PyTorch 正常
- [ ] dependency versions 与官方一致
- [ ] Llama-2-7B 正确加载

## Gate B：evaluation

- [ ] FP16 baseline 正常
- [ ] 官方 W2g64 checkpoint 可正常加载
- [ ] 官方 checkpoint 指标接近官方 README

## Gate C：Block-AP

- [ ] 4096 RedPajama samples
- [ ] context 2048
- [ ] W/s/z 正确训练
- [ ] 所有 blocks 完成
- [ ] Block-AP checkpoint 可重新加载

## Gate D：E2E-QP

- [ ] 4096 RedPajama samples
- [ ] context 4096
- [ ] effective batch 32
- [ ] only scale trainable
- [ ] 1 epoch 完成
- [ ] final checkpoint 可评测

## Gate E：结果

- [ ] Block-AP-only 指标已保存
- [ ] Block-AP + E2E-QP 指标已保存
- [ ] 与官方 W2g64 做了定量比较
- [ ] 实验 commit / environment / logs 可追踪

通过以上 Gate 后：

```text
Phase 1 完成
↓
开始 REAL-Prover BF16 → EfficientQAT W4/W2
```

---

# 24. 出错时的排查顺序

如果结果异常，不要同时修改多个变量。

严格按以下顺序：

### 1. 官方 checkpoint evaluation 是否正确？

如果不正确：

```text
先修 evaluation
```

### 2. FP16 baseline 是否正确？

如果不正确：

```text
检查 model / tokenizer / lm-eval
```

### 3. Block-AP 是否正常？

看：

```text
train loss
val loss
NaN
per-block progression
```

### 4. Block-AP checkpoint 能否 reload？

不能：

```text
先修 serialization / real_quant
```

### 5. E2E-QP 的 trainable params 是否只有 scales？

不是：

```text
先修 requires_grad / optimizer
```

### 6. 数据是否一致？

检查：

```text
RedPajama
train=4096
val=64
Block context=2048
E2E context=4096
seed
```

### 7. 最后才调整超参

第一轮复现中，不要因为结果不好立即：

- 改 LR
- 改 sample size
- 换 dataset
- 改 loss
- 改 quantizer

否则会失去“忠实复现”的意义。

---

# 25. 我们进入 REAL-Prover 前要得到的最终产物

```text
1. 一个可重复创建的 EfficientQAT 环境
2. 一套显式参数的复现脚本
3. Llama-2-7B FP baseline
4. 官方 W2g64 checkpoint evaluation
5. 自己训练的 Block-AP W2g64 checkpoint
6. 自己训练的 E2E-QP W2g64 checkpoint
7. 完整结果表
8. 训练时间 / 显存记录
9. Git commit history
10. 已知问题列表
```

这些构成后续：

```text
REAL-Prover
+
vanilla EfficientQAT
+
task-aware QAT
```

的可信 baseline。

---

# 26. 官方资料与版本基线

- EfficientQAT official repohttps://github.com/OpenGVLab/EfficientQAT
- ACL 2025 paperhttps://aclanthology.org/2025.acl-long.498/
- arXivhttps://arxiv.org/abs/2407.11062
- Official W2g64 checkpoint`ChenMnZ/Llama-2-7b-EfficientQAT-w2g64`
- RedPajama sample dataset used by code
  `togethercomputer/RedPajama-Data-1T-Sample`

---

## 当前执行顺序

```text
[1] clone upstream + 配置自己的 fork
        ↓
[2] 创建 conda 环境 + 安装 requirements
        ↓
[3] 下载 Llama-2-7B
        ↓
[4] R0：FP16 evaluation
        ↓
[5] R1：官方 W2g64 checkpoint evaluation
        ↓
[6] 确认评测环境正确
        ↓
[7] R3：Block-AP W2g64
        ↓
[8] 保存 Block-AP-only metrics/checkpoint
        ↓
[9] R4：E2E-QP
        ↓
[10] final evaluation + 对比官方结果
        ↓
[11] 整理 commit / metrics / known issues
        ↓
[12] 进入 REAL-Prover QAT
```

## R0–R4 自动运行脚本

已在 `EfficientQAT/scripts/run_r0_r4.sh` 添加有序总控脚本。先运行低成本 smoke：

```bash
cd /XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/tan/EFFICIENTQAT/EfficientQAT
bash scripts/run_r0_r4.sh smoke
```

Smoke 会加载 FP16 基座和官方 W2g64 checkpoint，各做一次短前向，并检查 Block-AP/E2E-QP 入口；不会启动训练或完整评测。完整实验按 **R0 → R1 → R2 → R3 → R4** 执行。R0/R1 会先对照 README 参考值做门控；若评测失败、缺指标或偏差超过默认容差（PPL ±1.0、平均准确率 ±5 个百分点），脚本停止，避免继续消耗数小时训练。可用 `R0_R1_PPL_TOLERANCE` 和 `R0_R1_ACC_TOLERANCE` 调整容差。

Smoke 通过且准备好长跑时，用 detached launcher 启动完整流程：

```bash
bash scripts/run_r0_r4.sh launch llama2_r0_r4
```

脚本会返回 PID。运行日志和状态在 `logs/r0_r4/llama2_r0_r4/`，checkpoint/结果在 `outputs/r0_r4/llama2_r0_r4/`；不要删除这些目录。可用 `tail -f logs/r0_r4/llama2_r0_r4/driver.log` 查看总控日志，`status.tsv` 查看每一步状态。也可在已连接的终端前台运行 `bash scripts/run_r0_r4.sh full llama2_r0_r4`。

---

# 27. R5（可选）：Instruction Tuning 实验复现与迁移验证

> 优先级：**完成 R0–R4、保存好 Block-AP-only checkpoint 之后**再做。R5 **不参与**现有 `scripts/run_r0_r4.sh` 的 smoke/full/launch 和 R0/R1 门控。与 REAL-Prover 的衔接价值很高，但不是证明 Section 4.1 正确复现的必要条件。

## 27.1 先分清：论文 Section 4.2 与我们现有复现模型不同

论文正式的 Section 4.2 / Table 3 使用 **Llama-1 7B / 13B**，在 **Alpaca** 上进行 instruction tuning，评估 **5-shot MMLU**；核心实验的模型、训练数据和指标均不同于 Section 4.1 的 **Llama-2/3 + RedPajama + zero-shot/PPL**。

| 对比项               | R0–R4：Section 4.1 主实验            | R5A：当前仓库可直接运行的扩展                   | R5B：严格对照论文 Section 4.2                  |
| -------------------- | ------------------------------------- | ----------------------------------------------- | ---------------------------------------------- |
| Backbone             | Llama-2-7B                            | **Llama-2-7B**                            | **Llama-1-7B**（另可做 13B）             |
| Block-AP             | RedPajama, W2g64                      | **复用 R3 的 RedPajama Block-AP**         | 需依论文核对并实现 Alpaca 的 Block-AP 数据路径 |
| E2E-QP               | RedPajama PT tokens                   | **Alpaca instruction/response**           | **Alpaca instruction/response**          |
| Loss                 | LM next-token CE                      | Response-only SFT CE                            | Response-only SFT CE                           |
| 可训练参数           | E2E 默认仅 scales                     | E2E 默认仅 scales                               | E2E 默认仅 scales                              |
| Evaluation           | WikiText2/C4 PPL、5 个 zero-shot 任务 | **5-shot MMLU**，可另记录 5-task          | **5-shot MMLU** 对照论文 Table 3         |
| 论文数值能否直接对齐 | 是                                    | **否：backbone 和 Block-AP 数据路径不同** | 在严格核对配置后才可以                         |

**当前建议：先做 R5A，不必为了这个扩展临时再下载并训练 Llama-1。** 如果今后以“论文所有主要表格严格复现”为目标，再独立开 R5B，不要把 R5A 的 Llama-2 MMLU 和论文 Llama-1 数值直接比较。

论文 Section 4.2 报告的训练设置：source_max_len=384、target_max_len=128、effective batch size=16、max_steps=10000；评估为 MMLU **5-shot**。论文原文参见 https://aclanthology.org/2025.acl-long.498.pdf 的 Section 4.2 和 Table 3。

## 27.2 Instruction Tuning 在本项目里意味着什么

Alpaca 样本包含 `instruction`、可选 `input` 和 `output`。数据加载器通过 `ALPACA_PROMPT_DICT` 将前两者格式化为 **source prompt**，将 `output` 作为 **target response**。

默认 `--train_on_source False`，collator 给 source labels 填入 `-100`，loss 主要监督回答 tokens（并按照代码处理 EOS）：

\[
\mathcal{L}_{\mathrm{SFT}}=-\sum_{t\in\mathrm{response}}\log p_\theta(y_t\mid x,y_{<t}).
\]

**注意**：这里的“Instruction Tuning”是训练数据与 loss 的种类，不代表全权重微调。在 `main_e2e_qp.py` 中仍然冻结 quantized backbone，仅将 `QuantLinear.scales` 设为可训练；不要误把这个实验当作 LoRA/QLoRA。

## 27.3 R5A：Llama-2-7B W2g64 + Alpaca E2E-QP

**关键：R5A 必须从 R3 的 Block-AP-only checkpoint 启动，不能从 R4 的 RedPajama E2E-QP 最终 checkpoint 接着训练。** 否则会多一个预先在 RedPajama 上 E2E 训练的阶段，无法干净比较“同一量化起点上的不同 E2E 数据”。

实验分支关系：

```text
FP16 Llama-2-7B
        |
     Block-AP (RedPajama)
        |
        +---- R4: E2E-QP (RedPajama) ----> zero-shot / PPL
        |
        +---- R5: E2E-QP (Alpaca) -------> 5-shot MMLU
```

数据：`tatsu-lab/alpaca`（https://huggingface.co/datasets/tatsu-lab/alpaca）。第一次运行在训练节点或本地 HF cache 准备好即可，不依赖其他新的数据格式。MMLU 评测数据由固定版本的 `lm-eval` task pipeline 加载。

### 运行命令（仓库已提供脚本，可直接复用）

仓库脚本：

`examples/e2e_qp/Llama-2-7b/w2g64-alpaca.sh`

**不要直接照抄脚本里的相对 Block-AP checkpoint 路径**；先确定 R3 实际保存到哪里，再将 `--quant_model_path` 和 `--output_dir` 改为自己机器上的真实路径。R0–R4 总控的 checkpoint 路径可能与原始仓库示例不同。

示例（手工将两个绝对路径替换为真实位置）：

```bash
cd "$QAT_ROOT/EfficientQAT"

export R3_BLOCK_AP_DIR=/absolute/path/to/R3/block_ap_checkpoint
export R5_OUTPUT_DIR="$QAT_ROOT/outputs/r5_llama2_w2g64_alpaca"
mkdir -p "$R5_OUTPUT_DIR" "$QAT_ROOT/logs"

CUDA_VISIBLE_DEVICES=0 python main_e2e_qp.py \
  --quant_model_path "$R3_BLOCK_AP_DIR" \
  --model_family Llama-2 \
  --wbits 2 \
  --group_size 64 \
  --learning_rate 2e-5 \
  --dataset alpaca \
  --dataset_format alpaca \
  --output_dir "$R5_OUTPUT_DIR" \
  --do_train True \
  --do_mmlu_eval True \
  --source_max_len 384 \
  --target_max_len 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --logging_steps 10 \
  --save_strategy steps \
  --evaluation_strategy steps \
  --max_steps 10000 \
  --eval_steps 2000 \
  --eval_dataset_size 16 \
  --bf16 \
  --data_seed 42 \
  --max_grad_norm 0.3 \
  --group_by_length \
  2>&1 | tee "$QAT_ROOT/logs/r5_llama2_w2g64_alpaca.log"
```

这是仓库 Llama-2 W2g64 Alpaca 示例的核心参数（路径与输出目录按本地环境替换）。**10,000 steps 是正式对照设置，不是快速 smoke test**。单卡 A800 可能需要较长运行时间；建议先单独短跑（如 `--max_steps 10` 并使用独立 debug 输出目录）确认数据、模型、梯度及 checkpoint 流程无误，正式运行再改回 10,000。不要把 smoke 结果当作论文指标。

如果实际 batch=16 OOM，可将 `--per_device_train_batch_size` 调为 4、`--gradient_accumulation_steps` 调为 4；这样有效 batch 仍为 16，但必须记录实际微批次配置和耗时。不要把更改配置后的结果误当完全同一运行条件。

## 27.4 R5 评测与结果记录

仓库 `main_e2e_qp.py` 在 `--do_mmlu_eval True` 时执行 `lm_eval.simple_evaluate(tasks=['mmlu'], num_fewshot=5)`；`lm_eval==0.4.2` 的任务格式与 prompt/选项 scoring 均由 harness 负责。训练时的 Alpaca prompt 则由 **本仓库** `datautils_e2e.py` 的 `ALPACA_PROMPT_DICT` 负责：两者不是同一套 prompt template。

R5 至少记录如下指标：

| Run | Backbone        | Quant | Block-AP 数据 | E2E 数据         | 可训练参数 | 5-shot MMLU | 备注                    |
| --- | --------------- | ----- | ------------- | ---------------- | ---------- | ----------: | ----------------------- |
| M0  | Llama-2-7B FP16 | FP16  | —            | —               | —         |        待测 | 相同 lm-eval 版本       |
| M1  | Llama-2-7B      | W2g64 | RedPajama     | —               | —         |        待测 | R3 Block-AP-only        |
| M2  | Llama-2-7B      | W2g64 | RedPajama     | RedPajama        | scales     |        待测 | R4 checkpoint           |
| M3  | Llama-2-7B      | W2g64 | RedPajama     | **Alpaca** | scales     |        待测 | **R5 checkpoint** |

对 M0–M3 最好在同一 `lm-eval==0.4.2`、同一 task config 下单独重新评测（不要混用不同 checkpoint/版本的已有数字），避免把模型本身差异错判成 Instruction Tuning 的收益。

R5A 的核心问题是：**在相同 W2 Block-AP 权重起点上，仅用 scale-only E2E-QP 配合 instruction-response SFT，是否能比 generic text E2E-QP 更好地保留/提升 instruction-following 与 MMLU 能力？**

⚠️ **R5A 并非论文 Table 3 的数值级严格复现**。论文 Table 3 的 Llama-1-7B W2g64 EfficientQAT 参考 MMLU 为 **32.6**（13B 为 **40.9**）；这些仅用于 R5B 的对照，不是 R5A Llama-2 的预期成绩。

## 27.5 R5B：以后真正严格复现 Table 3 时的额外工作

1. 独立获取 **Llama-1 7B**（需要时扩展到 13B），与当前 Llama-2 实验分开建目录/分支。
2. 重新确认 Section 4.2 所说“以 Alpaca 替换 RedPajama”的**两个阶段各自的数据设置**。当前仓库 `datautils_block.py` 只处理 `wikitext2 / c4 / redpajama`（其参数选择还列出一些未实现分支），**没有开箱即用的 Alpaca Block-AP loader**，需实现并做单元检查，不能默认为 RedPajama Block-AP + Alpaca E2E 就等于论文全配置。
3. 复现与 Table 3 同一 bit/group 的 Llama-1 checkpoint，并采用相同 SFT steps、prompt、loss mask、5-shot MMLU。
4. 保存 Table 3 可对齐的模型版本、dataset revision、随机种子与评测配置。

这部分消耗额外算力和工程时间，**不设为 REAL-Prover 阶段的进入条件**。

## 27.6 R5 完成标准及 Git 同步

- [ ] R3 Block-AP-only checkpoint 备份存在，路径经过核对
- [ ] Alpaca `instruction / input / output` 格式正确，source mask 为 `-100`
- [ ] E2E-QP 仅 `QuantLinear.scales` 可训练
- [ ] R5 smoke 2 steps 正常；正式运行 10,000 steps 结束（若时间受限，明确记录截断步数）
- [ ] 模型/训练日志/小型指标与 R4 目录分离，没有覆盖原结果
- [ ] 5-shot MMLU 的 harness 版本、任务配置与 results 已记录
- [ ] `R5_instruction_tuning.md`、R5 运行脚本和实验摘要已提交到自己的 fork

**推荐顺序：** `R0–R4（论文主量化复现） → R5A（现有 Llama-2 快速迁移验证） → REAL-Prover QAT`；如需严格复现全部论文实验，再独立插入 `R5B（Llama-1 Table 3）`


# 28. 本轮 R0–R5 自动流水线（2026-10-08，优先于旧命令）

R5A 提升为必需、高优先级；正式完成要求为 R0、R1、R2、R3、R4、R5 全部结束。
R3 只训练一次。R4 RedPajama 和 R5 Alpaca 均独立从 R3 Block-AP checkpoint 开始。
R5 正式参数沿用仓库示例：10,000 steps、batch 16、384/128 source/target、scale LR 2e-5。
R0/R3/R4/R5 均记录相同配置的 5-shot MMLU，保留各自产物；R5B 暂不纳入本轮。

## 28.1 低成本完整冒烟（不设置外部时限）

使用真实 Llama-2-7B 的全部 32 层，而非缩小架构：R0/R1 五任务各限 1 样本，WikiText2 1 段 128 tokens；
R2/R3 Block-AP 使用本地真实 RedPajama，train/val 各 2 段、128 tokens、1 epoch；R2/R4 E2E 各 1 step；
R5 用真实 Alpaca 数据、正确 instruction/response mask，4 train samples、2 steps、96/32 tokens。
R0/R3/R4/R5 的 57 个 MMLU 学科分别限 1 个 test 样本，仍用 5-shot prompt。
不使用 timeout 或外部两小时强制终止。通过少量真实数据、短序列和少量训练 steps 控制成本；每阶段失败仍会停止流水线。

通过标准：所有阶段成功；训练 loss 有限；仅 QuantLinear.scales 可训练；
R2/R4/R5 至少一个 scale 真正更新，所有冻结 tensors 的 SHA256 不变；
模型、tokenizer 和 trainer state 保存齐全；R2 checkpoint 重载前向有效；
R3/R4/R5 checkpoint 从磁盘重载完成 MMLU；四组 MMLU 指标有限。
微量数据指标只验证链路，不作为正式精度或论文复现结论。

```bash
cd "$QAT_ROOT/EfficientQAT"
source ../env/activate.sh
python scripts/prepare_r5_data.py
# 后台：冒烟成功后自动运行正式全流程；任一错误立即停止。
bash scripts/run_r0_r5.sh auto r0_r5_20261008
```

输出在 `logs/r0_r5/r0_r5_20261008_smoke/` 和 `outputs/r0_r5/r0_r5_20261008_smoke/`；
正式输出在相应 `_full/` 目录。总控日志为 `logs/r0_r5/r0_r5_20261008/auto.log`。
`SMOKE_OK` 仅在全部验收通过后写入，正式 `ALL_DONE` 仅在六组实验及 MMLU 全部结束后写入。
GPU 文件锁避免重复流水线并发占用同一卡。R0/R1 正式评测保留原论文参考门控。
正式运行时长不受两小时限制，R5 10,000 steps 可能明显长于 R0–R4。

## 28.2 Git 规则

每个代码修改批次前提交快照，修改后提交实现；失败修复也遵循相同顺序。
文档的仓库副本位于 `repro_notes/EFFICIENTQAT_REPRO_A800.md`，与项目根文档同步。
只提交源码、脚本、文档、小型验收摘要，不提交模型和数据。
本地提交身份为 haiyuntan；GitHub 推送能力需要网络连接与远端授权同时有效。


## 28.3 存储预算及 nohup 启动

冒烟和正式实验分别以 nohup 启动；总控等冒烟进程成功退出且 SMOKE_OK 存在后，才 nohup 启动正式进程。
数据准备单独 nohup 执行并等待完成，完成后再运行冒烟。
总控保存 prepare_data.pid、smoke.pid、full.pid 及各自 driver.log。

| 新增文件 | 冒烟 | 正式 |
| --- | ---: | ---: |
| R2 W4 与 R3 W2 Block-AP 模型 | 6–7 GiB | 6–7 GiB |
| R2/R4/R5 E2E 最终模型及 tokenizer | 9–12 GiB | 9–12 GiB |
| 中途 checkpoints 及 optimizer/RNG state | 不保存 | 15–22 GiB |
| 数据与 tokenization 缓存 | 通常 <1 GiB | 预算 10–40 GiB |
| 日志、指标 JSON、状态与审计 | 通常 <0.1 GiB | 通常 <1 GiB |
| 合计估算 | 16–20 GiB | 40–85 GiB |

合并保留冒烟与正式输出，建议预留 120 GiB 新增空间。实际大小依赖缓存与模型保存 dtype；
模型是完整 7B，冒烟减少样本不会缩小权重文件。R4/R5 复用 R3，不复制起点。
默认未启用磁盘激活卸载；若开启，Block-AP 两份 train/val FP16 激活会临时增加约 130 GiB。
已有模型和数据另计；共享磁盘空闲不等于个人 quota，启动前记录 df 与实际目录大小。


### 当前低成本配置（2026-10-08 更新）

| Run | 训练配置 | 保存后评测 |
| --- | --- | --- |
| R0 FP16 | 不训练，完整 7B | WikiText2 1×128 tokens，五任务各 1 样本，57 学科 MMLU 各 1 样本/5-shot |
| R1 官方 W2g64 | 不训练 | WikiText2 1×128 tokens，五任务各 1 样本 |
| R2 W4g128 | 全部 32 层 Block-AP；真实 RedPajama train=2、val=2、seq=128、batch=2、epoch=1；E2E 1 step/batch=1 | 重载最终模型，WikiText2 1×128 tokens、五任务各 1 样本 |
| R3 W2g64 | 全部 32 层 Block-AP；真实 RedPajama train=2、val=2、seq=128、batch=2、epoch=1 | 重载 Block-AP，WikiText2、五任务及 57 学科 MMLU/5-shot 小评测 |
| R4 W2g64 | 从 R3 起点独立做真实 RedPajama E2E，1 step/batch=1/seq=128 | 重载最终模型，WikiText2、五任务及 57 学科 MMLU/5-shot 小评测 |
| R5 W2g64 | 从 R3 起点独立做 Alpaca E2E；train=4、val=16、2 steps、batch=1、source/target=96/32 | 重载最终模型，WikiText2、五任务及 57 学科 MMLU/5-shot 小评测 |

所有 E2E 仅 scales 训练，bf16、grad accumulation=1；R2 scale LR=1e-5，R4/R5=2e-5。
Block-AP quant LR=1e-4；R2 weight LR=1e-5，R3 weight LR=2e-5。
微量数据验证完整训练/保存/重载/评测链路；成功后 nohup 启动原正式参数的 R0–R5。

### 本地数据加载（2026-10-08 更新）

R0–R5 统一使用 `local_datasets.py`：WikiText-2 和 MMLU 从现有 Arrow 缓存直接读取；PIQA、ARC、HellaSwag、WinoGrande、RedPajama 从本地 Parquet 读取；C4、Alpaca 从本地 JSON 读取。lm-eval 使用同一加载入口。运行启用 `HF_DATASETS_OFFLINE=1` 和 `HF_HUB_OFFLINE=1`；缺少文件立即报错，不回退在线下载。`prepare_r5_data.py` 在 nohup 冒烟开始前校验全部数据及评测任务。首次解析本地 Parquet/JSON 会生成 Arrow 缓存，空间计入原先的数据缓存预算。

### 实验分析文件同步到 GitHub

按照最新要求，实验期间不推送。`scripts/sync_experiment_reports.py --after-run <正式实验运行ID>` 用 nohup 等待该正式流水线的 `ALL_DONE` 标记，R0–R5 全部成功结束后才统一收集、提交并推送一次。失败或未完成时不会推送。同步包含配置、评测 JSON、训练指标、参数更新审计、状态和日志，存放在仓库 `experiment_reports/`；模型权重、数据集、缓存不上传。单个日志最多保留末尾 2 MiB；超过 2 MiB 的其他文本文件记录在 manifest 但不复制。推送失败记录错误，后续可手动重试；不会周期推送。网页端访问私有仓库仍需 GitHub 访问权限。

### 2026-10-08 冒烟修复与重新运行

修复 E2E-QP 的 LLaMA 特殊 token 初始化：保留 tokenizer 已有 eos/bos/unk，缺失时补齐，不再将缺失的模型 pad ID 当作 unk ID；同步 model.config.pad_token_id。清理之前 R0–R4/R0–R5 冒烟生成权重、评估结果、日志和专用缓存及报告快照，保留下载的基础模型、官方权重和本地数据集。新运行 r0_r5_20261008_v7 使用 nohup 从 R0 开始，冒烟通过后自动启动正式流水线；仅正式 R0–R5 全部完成后推送。
