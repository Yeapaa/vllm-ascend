"""
enable_thinking 字段测试（qwen、glm系列模型使用）- 流式
"""
from ...utility import request_helper as helper
from ...utility import assertion


def test_chat_template_kwargs_enable_thinking_true_stream(api_client, request):
    """enable_thinking=true，正常开启思考模式"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请用最简单的一句话介绍你是谁。"
        }],
        "chat_template_kwargs": {
            "enable_thinking": True
        },
        "stream": True,
        "max_tokens": 5120
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)


    # 校验点2：流式响应包含[DONE]
    assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason有效
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)

    # 校验点4：思考模式流式内容顺序（reasoning在前content在后，不并存；stop时必有content）
    assertion.assert_stream_thinking_content_order(response.text)


def test_chat_template_kwargs_enable_thinking_false_stream(api_client, request):
    """enable_thinking=false，关闭思考模式"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请用最简单的一句话介绍你是谁。"
        }],
        "chat_template_kwargs": {
            "enable_thinking": False
        },
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)


    # 校验点2：流式响应包含[DONE]
    assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason有效
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)

    # 校验点4：非思考模式（无reasoning_content/reasoning，存在content有值的chunk）
    assertion.assert_stream_no_reasoning_content(response.text)


# ==================== 异常场景 ====================

def test_chat_template_kwargs_enable_thinking_null_stream(api_client, request):
    """异常：enable_thinking为null"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "chat_template_kwargs": {
            "enable_thinking": None
        },
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)


def test_chat_template_kwargs_enable_thinking_string_stream(api_client, request):
    """异常：enable_thinking为字符串"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "chat_template_kwargs": {
            "enable_thinking": "true"
        },
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)