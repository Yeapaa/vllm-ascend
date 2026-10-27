import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_null(api_client, request, stream):
    """chat_template_kwargs为null，请求应正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "chat_template_kwargs": None,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)

    # 校验点3：finish_reason为stop或length
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_empty_object(api_client, request, stream):
    """chat_template_kwargs为空对象{}，请求应视为可选字段正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "chat_template_kwargs": {},
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)

    # 校验点3：finish_reason有效
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_with_add_generation_prompt(api_client, request, stream):
    """chat_template_kwargs设置add_generation_prompt参数，控制是否添加生成提示"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "chat_template_kwargs": {
            "add_generation_prompt": True
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_with_custom_system_prompt(api_client, request, stream):
    """chat_template_kwargs设置自定义系统提示词相关参数"""
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": "你是AI助手"
            },
            {
                "role": "user",
                "content": "你好"
            }
        ],
        "chat_template_kwargs": {
            "enable_system_prompt": True
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_date_params(api_client, request, stream):
    """chat_template_kwargs包含日期相关参数（某些模型支持动态日期）"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "今天是星期几"
        }],
        "chat_template_kwargs": {
            "date": "2025-04-01",
            "time": "10:00:00"
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_multiple_params(api_client, request, stream):
    """chat_template_kwargs包含多个有效参数组合"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请简单回答"
        }],
        "chat_template_kwargs": {
            "add_generation_prompt": True,
            "tools_prompt": "default",
            "custom_var": "custom_value"
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_with_tools_prompt(api_client, request, stream):
    """chat_template_kwargs设置tools_prompt参数，控制工具调用提示格式"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "需要查询天气"
        }],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "chat_template_kwargs": {
            "tools_prompt": "tool_instruction"
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_bos_token(api_client, request, stream):
    """chat_template_kwargs设置add_special_tokens或bos_token相关参数"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "chat_template_kwargs": {
            "add_special_tokens": True
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_template_kwargs_skip_special_tokens(api_client, request, stream):
    """chat_template_kwargs设置skip_special_tokens参数"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "chat_template_kwargs": {
            "skip_special_tokens": False
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)


def test_chat_template_kwargs_translation_content_no_prompt_leak(api_client, request):
    """关闭思考时，翻译结果不应泄漏输入提示词或聊天模板角色标记。"""
    system_content = (
        "不要添加任何解释性文字，只输出翻译结果。\n\n"
        "# Output Format\n"
        "仅输出翻译后的文本，不要包含任何前言、后语或注释。"
    )
    user_content = '{"target_language":"繁中(港)","source_text":"颅骨修补术"}'
    request_body = {
        "model": "auto",
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "chat_template_kwargs": {
            "enable_thinking": False,
            "thinking": False
        }
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    response_json = response.json()
    finish_reason = response_json["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)

    content = response_json["choices"][0]["message"].get("content", "") or ""
    assert isinstance(content, str), f"响应content应为字符串，实际类型为{type(content).__name__}"

    forbidden_contents = [
        system_content,
        user_content,
        "｜User｜",
        "｜Assistant｜"
    ]
    for forbidden_content in forbidden_contents:
        assert forbidden_content not in content, \
            f"响应content中不应包含请求提示词或角色标记：{forbidden_content}"
