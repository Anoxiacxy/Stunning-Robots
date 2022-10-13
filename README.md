# Stunning-Robots

这是一个离散的多智能体环境，以 PettingZoo 的 API 进行了封装。并提供了给予 ray-rllib 的训练接口。
 

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

## 环境描述

Stunning-Robots 是多智能体目标协同路径规划任务的强化学习环境。环境中允许存在 N 个智能体，每个智能体需要依次到达其对应的 M 个目标点。

**智能体**可以在地图上向上下左右移动，或者保持不动。环境中允许每个智能体有着不同的速度，我们假设所有智能体速度的最小公倍数为 L, 那么对于一个速度为 S 的智能体来说，它只允许在 L/S 的倍数轮次中移动。在不被允许移动的轮次，会有一个 action_mask 告知该智能体只能保持原地不动。

| Import             | from pettingzoo.classic import go_v5 |
|--------------------|--------------------------------------|
| Actions            | Discrete                             |
| Parallel API       | Yes                                  |
| Manual Control     | Yes                                  |
| Agents             | agents= ['agent_0', 'agent_1', ...]  |
| Agents             | any                                  |
| Action Shape       | Discrete(5)                          |
| Action Values      | Discrete(5)                          |
| Observation Shape  | (height, width, 7)                   |                                    
| Observation Values | [0, 1]                               |                                      

### 参数

自定义地图中的 .xlsx 文件需要有三个 sheet，分别命名为 "机器人配置"、"地图配置"、"位置坐标"。具体设置参考 [stunning_robots/maps/data1.xlsx](stunning_robots/maps/data1.xlsx)

将您自定义的地图放在 `stunning_robots/maps` 的文件夹中，以参数 `--map <your map config>` 的形式进行调用。
```python
import stunning_robots
# 默认地图参数
env = stunning_robots.parallel_env(**stunning_robots.DEFAULT_CONFIG) 
# 自定义地图参数
env = stunning_robots.parallel_env(**stunning_robots.load_map_config("path/to/custom/maps/data.xlsx")) 
```
### 动作空间
每个智能体都有以下 5 个可能的行动，描述为 `Discrete(5)`
```python
HOLD, LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3, 4
```

### 观察空间

每个智能体的观察空间都是一个字典类型（Dict），其中包含了`'obs'`元素，是一个形状为`(h, w, 7)`的 0-1 数组，描述了地图上的墙体信息，自身位置信息，其他智能体的位置信息，自身当前目标信息，其他智能体当前目标信息，还有自身所有目标位置信息，其他智能体所有目标位置信息。

字典中的另一个元素是`'action_mask'`同样是一个形状为`(5)`的 0-1 数组，描述了当前智能体可以做的合法动作（0-不合法，1-合法）。 具体而言，当轮到智能体行动的时候，`'action_mask'` 是全1的数组，而当智能体不能行动时，只有 `HOLD` 行动是 1，而其他移动操作都是 0。

### 奖励

默认的奖励设置为：

- `GOAL_REWARD`：智能体达到一个目标点的奖励，默认为 +100
- `WALL_COLLISION_REWARD`：智能体撞墙的奖励，默认为 -1
- `PEER_COLLISION_REWARD`：智能体和其他智能体相撞的奖励，默认为 -1
- `OUT_OF_BOUNDARY_REWARD`：智能体超出地图边界的奖励，默认为 -1
- `POTENTIAL_REWARD`：智能体的势能变化奖励，默认为 +1.5
- `TIME_PUNISHMENT`：如果智能体没有到达目标点，每一步的时间惩罚，默认为 -2.5

您可以在 [stunning_robots/config.py](stunning_robots/config.py) 内自行修改。

## 运行

### 推荐训练命令

```bash
python run_ppo.py train --policy speed --map data1.xlsx --step 500 --render --episode 200
python run_ppo.py test  --policy speed --map data1.xlsx --step 500 --record
```

### 所有支持的参数

```
usage: run_ppo.py <phase> [-h] [-r] [-R] [-p POLICY] [-c CHECKPOINT] [-m MAP] [-M MANUAL] 
                          [-s STEP] [-e EPISODE] [--checkpoint_every CHECKPOINT_EVERY] 
```
参数详细说明：
```
Train PPO to play in stunning-robots.

positional arguments:
  <phase>               'train' or 'test'

optional arguments:
  -h, --help            show this help message and exit
  -r, --record          whether to record the test result
  -R, --render          whether to render during the train
  -p POLICY, --policy POLICY
                        'parallel', 'single', 'speed'
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        load which checkpoint
  -m MAP, --map MAP     your custom map xlsx file
  -M MANUAL, --manual MANUAL
                        manually operate one specified agent
  -s STEP, --step STEP  the number of steps of an episode
  -e EPISODE, --episode EPISODE
                        the number of episodes to train
  --checkpoint_every CHECKPOINT_EVERY
```
一些使用的例子：
```bash
# 默认训练参数
python run_ppo.py train
# 指定使用 data1.xlsx 配置文件 
python run_ppo.py train -m data1.xlsx
# 指定环境最大步数为 500 
python run_ppo.py train -s 500
# 指定总共训练 500 轮
python run_ppo.py train -e 500
# 使用 ppo_parallel 模型训练，并加载第 165 个 checkpoint
python run_ppo.py train -p parallel -c 165
# 训练的时候开启渲染模式
python run_ppo.py train --render
# 每训练 10 轮就保存一次模型
python run_ppo.py train --checkpoint_every 10

# 默认测试参数
python run_ppo.py test
# 指定使用 data1.xlsx 配置文件 
python run_ppo.py test -m data1.xlsx
# 指定环境最大步数为 500 
python run_ppo.py test -s 500
# 录制
python run_ppo.py test --record
# 使用 ppo_speed 模型测试，并加载第 100 个 checkpoint
python run_ppo.py test -p speed -c 100
# 测试的时候可以使用键盘操控第 3 个智能体（编号从 0 开始）
python run_ppo.py test --manual 3

```

### 策略分配方式

- `single`：单一策略控制多智能体，所有的智能体使用同样的策略控制，策略的泛化性能可能更好。模型保存在 `ppo_single-xxxx` 文件夹下。（注：旧版训练得到的模型保存在`ppo-xxxx` 下，可以直接修改文件名以加载）

```bash
python run_ppo.py train -p single
python run_ppo.py test -p single
```
- `parallel`：多个策略控制多智能体，策略与智能体一一对应，每个智能体背后使用同构但是不同参数的策略网络进行控制。模型保存在 `ppo_parallel-xxxx` 文件夹下。

```bash
python run_ppo.py train -p parallel
python run_ppo.py test -p parallel
```

- `speed`：相同速度的智能体使用同一个策略——不同速度的智能体使用不同的策略。模型保存在 `ppo_speed-xxxx` 文件夹下。（注：例如在 `data1.xlsx` 配置文件中，所有的智能体一共有三种可能的速度，分别是 1、2、3，那么我们只会存在三种策略，然后用着三种策略控制对应速度的智能体）

```bash
python run_ppo.py train -p speed
python run_ppo.py test -p speed
```


## 手动操作

这里提供了手动操作的策略接口，可以使用键盘 WASD 对某一个智能体进行操控

```bash
# 操控 0 号智能体
python run_ppo.py test --record --map data1.xlsx --step 5000 --policy speed --manual 0
```