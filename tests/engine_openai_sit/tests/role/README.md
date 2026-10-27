# Role 角色参数测试套件

## 简介

本测试套件用于验证推理引擎中 `messages` 数组的 `role` 字段处理逻辑。

## Role 字段说明

- **含义**: 标识消息的发送者角色
- **有效值**: 
  - `system`: 系统指令/提示词
  - `user`: 用户输入
  - `assistant`: 助手回复
- **必需性**: 必需字段
- **大小写**: 区分大小写（`user` ≠ `User`）

## 测试文件说明

| 文件 | 说明 | 用例数 |
|------|------|--------|
| `test_role_normal.py` | 正常场景测试 | 8个 |
| `test_role_abnormal.py` | 异常值测试 | 14个 |
| `test_role_effect.py` | 效果验证 | 5个 |

## 测试覆盖场景

### 正常场景
- 单条user消息
- system + user 组合
- 多轮对话: user-assistant-user 交替
- 完整对话: system-user-assistant-user
- 仅system（边缘情况）
- 多条system（覆盖行为）
- 长历史对话（20轮）
- 空内容消息

### 异常场景
- 无效角色值: `unknown`
- 缺少role字段
- null值
- 整数类型
- 空字符串
- 大小写敏感: `User`、`SYSTEM`

### 效果验证
- system指令是否影响输出
- 多轮对话上下文记忆
- 多条system覆盖行为
- assistant消息作为上下文的传递
- 超长历史处理

## 运行测试

```bash
# 运行所有role测试
pytest engine_func_test_robot/tests/role/ -v

# 运行特定测试
pytest engine_func_test_robot/tests/role/test_role_normal.py -v

# 运行流式测试
pytest engine_func_test_robot/tests/role/ -v -k "stream"

# 运行非流式测试
pytest engine_func_test_robot/tests/role/ -v -k "non_stream"
```

## 注意事项

1. 标准OpenAI API要求消息必须以user结束
2. 某些实现不允许纯system对话（必须有user消息）
3. 系统指令的影响程度因模型而异
4. 长历史可能触发上下文截断
5. role字段区分大小写是标准要求
