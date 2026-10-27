# Temperature 测试套件

## 简介

本测试套件用于验证推理引擎中 `temperature` 采样参数的处理逻辑。

## Temperature 参数说明

- **含义**: 控制采样时的随机性/创造性程度
- **范围**: 0.0 ~ 2.0（常见实现）
- **默认值**: 1.0
- **作用原理**: 对logits除以temperature后再softmax
  - temperature=0: 完全贪婪解码（确定性）
  - temperature<1: 降低随机性，输出更保守
  - temperature=1: 标准采样
  - temperature>1: 增加随机性，输出更创造性

## 测试文件说明

| 文件 | 说明 | 用例数 |
|------|------|--------|
| `test_temperature_normal.py` | 正常值测试 | 7个 |
| `test_temperature_abnormal.py` | 异常值测试 | 12个 |
| `test_temperature_boundary.py` | 边界值测试 | 7个 |
| `test_temperature_effect.py` | 效果验证 | 6个 |

## 测试覆盖场景

### 正常场景
- 常规取值: 0.0, 0.1, 0.5, 0.8, 1.0, 1.5, 2.0
- 确定性输出: temperature=0.0
- 创造性输出: temperature=2.0
- 参数组合: 与top_p、n组合
- 默认值: 不设置时使用默认1.0

### 异常场景
- 负数: temperature < 0
- 超出上限: temperature > 2.0
- 类型错误: 字符串、null、数组、对象
- 非法格式

### 边界场景
- 0.0边界
- 2.0边界
- 极小正数
- 极大接近上限
- 高精度小数
- 不设置使用默认值

### 效果验证
- 相同输入多次请求结果一致性（temperature=0）
- 低温保守性
- 高温创造性
- 与n>1组合的多样性
- 与top_p的协同效果

## 运行测试

```bash
# 运行所有temperature测试
pytest engine_func_test_robot/tests/temperature/ -v

# 运行特定测试
pytest engine_func_test_robot/tests/temperature/test_temperature_normal.py -v

# 运行流式测试
pytest engine_func_test_robot/tests/temperature/ -v -k "stream"

# 运行非流式测试
pytest engine_func_test_robot/tests/temperature/ -v -k "non_stream"
```

## 注意事项

1. 某些模型的temperature上限可能不同（如部分支持到2.5或3.0）
2. temperature=0时依赖top_p=0才能实现完全贪婪
3. 流式和非流式在错误处理上返回方式不同
