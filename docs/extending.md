# 新增控制器、网络与模型

先复制一个小规模实验配置，固定路径、种子和后端，再一次改变一个组件。默认配置使用 `research.controllers.ddat`、`research.networks.ddat`、`research.models.uuv`，训练算法仍为 PPO。

## 配置与构造约定

`experiments/run.py` 的 JSON 采用 `name`、`mode`、`arguments` 结构。`mode` 为 `train` 或 `validate`；组件字段放在训练配置的 `arguments` 内，训练后随 checkpoint 保存。验证通过 checkpoint 恢复组件，无需手动复制模块名。

```json
{
  "name": "ddat_uuv_wide_smoke_s1027",
  "mode": "train",
  "arguments": {
    "device": "cpu",
    "backend": "admm",
    "envs": 2,
    "eval_envs": 2,
    "total_steps": 64,
    "rollout": 8,
    "seed": 1027,
    "controller_module": "research.controllers.ddat",
    "network_module": "research.networks.ddat",
    "model_module": "research.models.uuv",
    "controller_kwargs": {},
    "network_kwargs": {
      "hidden_sizes": [128, 128],
      "encoder_sizes": [64, 32]
    }
  }
}
```

将其保存为 `experiments/configs/ddat_uuv_wide_smoke_s1027.json`，即可运行；仓库也提供了 [小网络完整配置](../experiments/configs/ddat_uuv_small_network_smoke_s1027.json)，可直接复制修改。

```bash
python3 experiments/run.py experiments/configs/ddat_uuv_wide_smoke_s1027.json --dry-run
python3 experiments/run.py experiments/configs/ddat_uuv_wide_smoke_s1027.json
```

默认网络宽度是 `hidden_sizes=[256,256]`、`encoder_sizes=[128,64]`，省略字段时保留原 checkpoint 结构。网络结构变化后应使用新 checkpoint。启动器把两个 kwargs 对象序列化为 CLI 的 JSON 字符串；直接调用时可以写 `--network-kwargs '{"hidden_sizes":[128,128]}'`。

`controller_kwargs` 只存插件额外参数；`seed`、`horizon`、`backend`、`randomize_plant` 等环境参数放在普通 `arguments` 中，与受管理参数重名会报错。`network_kwargs` 不能覆盖 `observation_dim`；观测宽度由环境提供。设备和批量大小由工厂显式管理，验证时的种子等覆盖项最终生效。模型只接收 `device` 构造参数，当前没有 `model_kwargs`。

## 控制器：为动作加入代价

新增 `research/controllers/action_penalty.py`，导出固定名称 `ExperimentDDATPathEnv`。下面的例子保留原观测和终止逻辑，只为活动回合增加策略动作平方代价：

```python
import torch

from research.controllers.ddat import ExperimentDDATPathEnv as BaseEnv


class ExperimentDDATPathEnv(BaseEnv):
    def __init__(self, *args, action_penalty=0.02, **kwargs):
        if action_penalty < 0:
            raise ValueError("action_penalty must be nonnegative")
        self.action_penalty = float(action_penalty)
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def step(self, action):
        active = ~self.finished
        observation, reward, done, info = super().step(action)
        cost = action.clamp(-1, 1).square().sum(-1)
        reward = reward - self.action_penalty * cost * active
        return observation, reward, done, info
```

配置改为 `"controller_module": "research.controllers.action_penalty"` 和 `"controller_kwargs": {"action_penalty": 0.02}`。环境必须接受公共工厂传入的参数，并保留 `reset(path_ids, mask=None)`、`observe()`、`step(action)` 和训练/评估使用的状态、MPC、统计字段。继承默认环境可保留这些约定。

动作形状为 `[batch_size, 3]`，语义是 DDAT 的推进力及俯仰/偏航角速度修正；它不是直接施加给模型的三维物理执行器控制量。`step` 返回 `(observation, reward, done, info)`。如需改变观测，更新 `observation_size`，同步实现兼容网络，并使用新的方案名和 checkpoint。

## 网络：替换预览编码器

内置宽度参数足够时，直接使用前面的 `network_kwargs`。需要研究新的结构时，可新增 `research/networks/wide_encoder.py`，导出 `ActorCritic`：

```python
from torch import nn

from research.networks.ddat import ActorCritic as BaseActorCritic


class ActorCritic(BaseActorCritic):
    def __init__(self, observation_dim, encoder_width=96):
        super().__init__(observation_dim)
        self.feedforward_encoder = nn.Sequential(
            nn.Linear(self.preview_size, encoder_width),
            nn.ReLU(),
            nn.Linear(encoder_width, 64),
            nn.ReLU(),
        )
```

配置使用 `"network_module": "research.networks.wide_encoder"`、`"network_kwargs": {"encoder_width": 96}`。构造函数接受观测宽度；实例是 `torch.nn.Module`，支持 `.to(device)`、`state_dict()` 和 `load_state_dict()`。策略接口保持如下含义：

| 方法 | 返回值 |
| --- | --- |
| `deterministic(observation)` | `[batch, 3]` 的确定性动作，范围 `[-1,1]` |
| `value(observation)` | `[batch]` 的状态价值 |
| `sample(observation)` | 动作、对应 log probability、价值 |
| `log_prob(observation, action)` | `[batch]` 的动作 log probability |
| `distribution(observation)` | 支持 `.entropy()` 的分布；PPO 对动作维求和计算熵项 |

修改结构后，检查采样与 `log_prob` 的动作变换一致，并验证新 checkpoint 保存/加载后的确定性输出一致。

## 模型：改变标称阻尼

新增 `research/models/high_drag.py`，导出 `DynamicsModel`。保持现有动力学接口最直接的方式是继承默认批量模型：

```python
from simulator.batch_mpc import BatchedUUVModel


class DynamicsModel(BatchedUUVModel):
    def __init__(self, device):
        super().__init__(device)
        self.damping = self.damping * 1.05
```

配置使用 `"model_module": "research.models.high_drag"`，先选择 `"backend": "admm"`。所选实例用于环境和 MPC 的预测，因此这个例子改变共同标称阻尼；单独研究真实艇体与预测模型之间的失配，需要另行设计两者的参数差异，不能把该配置当作仅改变真实艇体。

模型需要保留 `device`、`dtype`、`mass`、`inverse_mass`、`damping`、`elevator`、`rudder`、`control_scale`、`lower`、`upper`，以及 `tensor`、`derivative`、`rk4`、`linearize` 接口。状态、控制、扰动和海流的形状与含义以 `BatchedUUVModel` 为准：10 维状态、3 维实际执行器输入、5 维机体系加速度扰动、3 维 NED 海流；`linearize` 接受归一化控制，`derivative`/`rk4` 接受物理控制。

现有 RTI 后端使用固定的符号模型，因此环境会拒绝同时使用自定义 `model_factory` 与 `rti_cpu`/`rti_cuda`。使用新动力学的 RTI 实验前必须扩展并验证符号模型与数值模型的一致性。该插件入口也不支持直接替换为任意维度或不同坐标定义的机器人。

路径的 `feasible_only` 筛选仍依据原标称模型。选择自定义模型不会重算这套筛选标准；新动力学的可行性需要独立评估，不能把既有筛选结果当作对该模型的认证。

## 验证与比较

先运行 [测试说明](../tests/README.md) 中的基础检查，再为实际修改验证有意义的行为：奖励改变方向、网络输出及保存/加载一致性、动力学导数与离散积分、MPC 使用的模型实例。用小规模实验确认训练与 checkpoint 验证链路后，再扩大训练预算。

对照实验保留相同路径、种子、海流和随机艇体参数，输出分别写到独立运行目录。报告记录模块名、kwargs、后端、数据划分、实际评估回合和失败情况。配置可以组合多个插件，但定位贡献时应先分别做消融。
