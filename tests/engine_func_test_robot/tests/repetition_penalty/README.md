# Repetition Penalty 测试套件

## 简介

本测试套件用于验证推理引擎中 `repetition_penalty` 重复惩罚参数的处理逻辑。该参数在vLLM等开源推理引擎中广泛使用。

## Repetition Penalty 参数说明

- **含义**: 对生成过的token施加乘法惩罚
- **范围**: 1.0 ~ 2.0
- **默认值**: 1.0
- **作用原理**: 
  - 1.0: 无重复惩罚
  - >1.0: 已生成token的概率被除以该值（降低概率）
  - 值越大，重复概率越低
- **计算方法**: `score = score / repetition_penalty`

## 与 Frequency/Presence Penalty 的区别

| 参数 | 计算方式 | 适用场景 |
|------|----------|----------|
| repetition_penalty | 乘法惩罚，对已生成token | vLLM等推理引擎 |
| frequency_penalty | 加法惩罚，根据出现次数 | OpenAI API |
| presence_penalty | 加法惩罚，二元判断 | OpenAI API |

## 测试文件说明

| 文件 | 说明 | 用例数 |
|------|------|--------|
| `test_repetition_penalty_normal.py` | 正常值测试 | 9个 |
| `test_repetition_penalty_abnormal.py` | 异常值测试 | 14个 |

## 测试覆盖场景

### 正常场景
- 常规取值: 1.0, 1.05, 1.1, 1.2, 1.5, 2.0
- 无效果: 1.0
- 适度惩罚: 1.2
- 强烈抑制: 2.0
- 多惩罚组合: 与frequency_penalty、presence_penalty、temperature
- 所有惩罚同时使用

### 异常场景
- 小于1.0: <1.0（非法）
- 等于0: 0.0（非法）
- 负数（非法）
- 类型错误: 字符串、null、数组、对象

## 运行测试

```bash
# 运行所有repetition_penalty测试
pytest engine_func_test_robot/tests/repetition_penalty/ -v

# 运行特定测试
pytest engine_func_test_robot/tests/repetition_penalty/test_repetition_penalty_normal.py -v

# 运行流式测试
pytest engine_func_test_robot/tests/repetition_penalty/ -v -k "stream"

# 运行非流式测试
pytest engine_func_test_robot/tests/repetition_penalty/ -v -k "non_stream"
```

## 注意事项

1. repetition_penalty是vLLM特有的参数，OpenAI API不直接支持
2. 与frequency_penalty/presence_penalty可以共存，效果叠加
3. 值小于1.0在数学上会导致重复token概率增加，通常被禁止
