# DCP MoE 测试交接

测试版本：`feat/dynamic-cp-moe`，commit `fd9fa1c1`。本文记录本次实际执行的验证，不代表所有配置均已覆盖。

## 已验证的结果

### 1. CPU 多进程测试：5 passed

`tests/test_dynamic_cp_moe.py`：使用真实 Megatron router，混合 CP1/2/4，与相同 microbatch 的 CP1 参考比较。

- `aux_loss`、`seq_aux_loss` × per-sample、per-token，共四种组合通过。
- 辅助 loss 一致：`rtol=1e-5`；对 router logits 的梯度一致：`rtol=1e-4, atol=1e-7`。
- Padding 位置辅助梯度为零；TP×动态 CP 通信组测试通过。
- 这是 router 辅助项测试，主 loss 置零，不是完整模型梯度测试。

### 2. Granite EP4：四次单步 SFT 训练全部完成

模型：`/public/home/xinxin005/syk/model/ibm-granite/granite-3.1-3b-a800m-base`。
4×A40，TP1 / PP1 / EP4，BF16，full recompute，distributed optimizer，lr=1e-6，global batch=8，每卡预算512。
输入长度：2000、1800、1600、1016、1000、504、496、488；各对照输入相同。DCP 实际使用 CP1/2/4。

| 检查项 | 参考 | DCP | 相对差异 |
| --- | ---: | ---: | ---: |
| 主 loss，aux coeff=0，CP4 对照（gpu4） | 2.160052776 | 2.159112692 | 0.04352% |
| 主 loss，aux coeff=0.01，CP1 对照（gpu5 的0–3卡） | 2.160004616 | 2.160139084 | 0.006225% |
| 上一组 load-balancing loss 日志值 | 1.44766188 | 1.44767869 | 0.001161% |

开启辅助项时使用 CP1 参考，是为保持每个 microbatch 一条样本、全局8个 microbatch，避免固定 CP4 packing 改变辅助 loss 的统计分组。
结论：上述配置可完成 forward/backward/optimizer step，主 loss 和辅助 loss 数值接近。本轮按用户要求到此结束，不追加梯度或多步测试。

## 适配与复现入口

- Slime：本工作树；Megatron：`/public/home/xinxin005/syk/slime-env-setup/Megatron-LM-dcp`。
- 外部适配包：`/public/home/xinxin005/syk/slime-env-setup/dcp-e2e`，其中 `granite_bridge.py` 注册 HF loader，`run_moe.py` / `dcp_moe_probe.py` 在 driver / actor 中加载它。
- Granite 支持在外部适配包中，不在 Slime 原生模型注册表中。适配把缩放折进权重并解除 embedding 权重共享；本次验证的是同一适配模型的并行布局一致性，不是原生 HF 训练等价性。
- 启动脚本：`/public/home/xinxin005/syk/slime/benchmarks/context_parallel/run_granite_ep4_check.sh`。`main PREFIX` 跑 CP4/DCP；`aux PREFIX` 跑 CP1/DCP。必须使用新 PREFIX，默认 GPU0–3，可通过 `SLIME_TEST_GPUS` 指定。无需重新运行本轮已完成测试。
- 原始日志：`/public/home/xinxin005/syk/slime-env-setup/dcp-e2e/out/granite-ep4-check-v1-{main,aux}-{fixed,dcp}/train.log`。
- 汇总：`/public/home/xinxin005/syk/slime/benchmarks/context_parallel/granite-ep4-check-v1-results.json`。

## 撤回的测试及范围边界

曾比较整块 `main_grad` buffer，但 distributed optimizer reduce-scatter 后只有本 rank 归属的切片具有最终归约值；该比较包含非归属区域，方法无效。此前42%/54%的梯度差异结论撤回，不应作为实现错误的证据。按用户要求已删除该额外测试、hook 和梯度/参数抽样文件；保留有效 loss 日志。不据此声称完整模型梯度已对齐。

本轮不验证 PP、其他未支持组合、多步收敛、原生 HF 等价性或吞吐。旧 dense 吞吐测试未使用本分支的新 MoE 实现，不能代替 MoE 性能测试。OLMoE 只登记，未运行。

## 后续已授权的性能队列

用户已授权启动 Granite 随机长度吞吐对照：gpu4 EP4/4卡、gpu5 EP8/8卡，TP1/PP1，固定CP4分别为DP1/DP2，对比DCP；三个范围1K–8K/16K/32K，预算2048/4096/8192。每配置25步，前2步预热，global batch32，aux coeff0.01，SwanLab离线。没有追加梯度测试。

启动脚本：`/public/home/xinxin005/syk/slime/benchmarks/context_parallel/run_granite_speed.sh`。
tmux：gpu4 `granite-speed-ep4`；gpu5 `granite-speed-ep8`。
输出：`/public/home/xinxin005/syk/slime/benchmarks/context_parallel/runs/`下的`granite-speed-gpu4-v2/`和`granite-speed-gpu5-v5/`，查看`status.txt`确认进度，不能把启动视为已完成。
训练使用独立`granite_runtime_v1`快照，不直接读取本工作树；Claude后续修改分支不会改变这次运行。计时使用monotonic clock，记录每步实际CP分配、峰值显存、loss和吞吐。
