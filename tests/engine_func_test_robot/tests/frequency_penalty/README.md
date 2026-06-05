# Frequency Penalty 测试套件

## 简介

本测试套件用于验证推理引擎中 `frequency_penalty` 频率惩罚参数的处理逻辑。

## Frequency Penalty 参数说明

- **含义**: 根据token在生成文本中出现的**频率**降低其概率
- **范围**: -2.0 ~ 2.0
- **默认值**: 0.0
- **作用原理**: 
  - 正值: 高频出现的token概率降低，减少重复用词
  - 负值: 高频出现的token概率增加，鼓励重复
  - 0.0: 无频率惩罚

## 与 Presence Penalty 的区别

| 参数 | 计算方式 | 效果 |
|------|----------|------|
| frequency_penalty | 根据token出现次数累加惩罚 | 多次出现受重罚 |
| presence_penalty | token只要出现过就施加固定惩罚 | 出现即受罚 |

## 测试文件说明

| 文件 | 说明 | 用例数 |
|------|------|--------|
| `test_frequency_penalty_normal.py` | 正常值测试 | 8个 |
| `test_frequency_penalty_abnormal.py` | 异常值测试 | 12个 |

## 测试覆盖场景

### 正常场景
- 常规取值: -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0
- 无效果: 0.0
- 正值减少重复
- 负值增加重复
- 多惩罚组合: 与presence_penalty、repetition_penalty组合
- 所有惩罚同时使用

### 异常场景
- 超上限: >2.0
- 超下限: <-2.0
- 类型错误: 字符串、null、数组、对象

## 运行测试

```bash
# 运行所有frequency_penalty测试
pytest engine_func_test_robot/tests/frequency_penalty/ -v

# 运行特定测试
pytest engine_func_test_robot/tests/frequency_penalty/test_frequency_penalty_normal.py -v

# 运行流式测试
pytest engine_func_test_robot/tests/frequency_penalty/ -v -k "stream"

# 运行非流式测试
pytest engine_func_test_robot/tests/frequency_penalty/ -v -k "non_stream"
```

## 注意事项

1. 部分实现可能不接受负值
2. 流式和非流式错误返回方式可能不同
3. 与presence_penalty有细微区别，需分别测试
