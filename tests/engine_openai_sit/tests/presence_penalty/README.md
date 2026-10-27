# Presence Penalty 测试套件

## 简介

本测试套件用于验证推理引擎中 `presence_penalty` 存在惩罚参数的处理逻辑。

## Presence Penalty 参数说明

- **含义**: 如果token已经在生成文本中**出现过**，则降低其概率
- **范围**: -2.0 ~ 2.0
- **默认值**: 0.0
- **作用原理**: 
  - 正值: 出现过的token概率降低，鼓励讨论新话题
  - 负值: 出现过的token概率增加，鼓励停留在当前话题
  - 0.0: 无存在惩罚

## 与 Frequency Penalty 的区别

| 参数 | 计算方式 | 效果 |
|------|----------|------|
| presence_penalty | 只要出现过的token都受固定惩罚 | 鼓励话题切换 |
| frequency_penalty | 根据出现次数累加惩罚 | 减少重复用词 |

## 测试文件说明

| 文件 | 说明 | 用例数 |
|------|------|--------|
| `test_presence_penalty_normal.py` | 正常值测试 | 8个 |
| `test_presence_penalty_abnormal.py` | 异常值测试 | 12个 |

## 测试覆盖场景

### 正常场景
- 常规取值: -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0
- 无效果: 0.0
- 正值: 鼓励新话题
- 负值: 鼓励停留话题
- 极值: -2.0和2.0边界
- 多惩罚组合

### 异常场景
- 超上限: >2.0
- 超下限: <-2.0
- 类型错误: 字符串、null、数组、对象

## 运行测试

```bash
# 运行所有presence_penalty测试
pytest engine_func_test_robot/tests/presence_penalty/ -v

# 运行特定测试
pytest engine_func_test_robot/tests/presence_penalty/test_presence_penalty_normal.py -v

# 运行流式测试
pytest engine_func_test_robot/tests/presence_penalty/ -v -k "stream"

# 运行非流式测试
pytest engine_func_test_robot/tests/presence_penalty/ -v -k "non_stream"
```

## 注意事项

1. presence_penalty与frequency_penalty经常一起使用，效果互补
2. 正值会鼓励模型切换话题，避免长篇大论在一个点上
3. 部分实现可能对范围校验不严格
