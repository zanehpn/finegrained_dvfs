# MIDAS DVFS code

MIDAS 论文（DAC 2027）的 DVFS 建模、在线控制器、实验和出图代码。这里是 `/shared/s3/zqy/dac2027_p1` 的快照，那边仍是主工作目录。

思路和现状见 `PROPOSAL.md`。

## scripts/
| 文件 | 作用 |
|---|---|
| `qwen_first_order_v1.py`、`qwen_first_order_v3.py` | 校准后的一阶请求模型 |
| `llama_first_order_v2.py`、`llama_bitmod_first_order_v1.py`、`qwen32b_first_order_v1.py` | 其他模型和数据类型 |
| `cal_first_order_v1.py` | 一阶模型校准 |
| `online_planner_v1.py` | T3 因果在线控制器（`own_twin`、`run_online`） |
| `online_eval_v1.py`、`serve_models_v1.py` | 工作负载安装和连续批处理图 |
| `real_slo_v1.py`、`real_trace_v1.py`、`real_guard_v1.py` | 主实验：Azure trace、MLPerf 目标、同违例率比较 |
| `online_robust_v1.py`、`coarse_granularity_v1.py`、`subblock_v1.py`、`spread_v1.py`、`joint_alloc_v3.py` | 鲁棒性、单电压域、子块、bimodal、联合分配 |
| `motivation_precision_v1.py`、`t2_request_v1.py` | 45 组精度实验、T2 请求级评估 |
| `paper_figures_v*.py` | 论文图 |
| 其余 | 上述脚本的依赖 |

## run/
实验启动脚本，给出了环境变量和 worker 数。当前主实验是 `real_b2.sh`，guard 扫描是 `real_g.sh` 和 `real_g2.sh`。

## data/
`first_order_cal_final.json`：冻结的校准参数，对应环境变量 `QWEN_CAL`。

## paper/
当前论文章节、新增参考文献和预览 PDF，Azure 数字仍是 TBD。

## variants/att2/
双 ATT 版本的代码差异和全套实验脚本，完整目录在 `/shared/s3/zqy/dac2027_p1_att2`。

## 运行限制
- 脚本里写死了 `/shared/s3/zqy/dac2027_p1` 和 `/shared/s3/zqy/DSE/npu_mac` 的路径，换位置要先改路径。
- 还依赖 `DSE/npu_mac` 下的校准数据、精度方案、Azure trace 和 `scripts/` 里的其他模块，这里没有复制。
- `ON_SLOHEDGE=1` 打开延迟保护，默认关闭，论文结果未使用。
# finegrained_dvfs
