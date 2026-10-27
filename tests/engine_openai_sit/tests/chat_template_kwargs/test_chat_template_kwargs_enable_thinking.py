"""
enable_thinking 字段测试（qwen、glm系列模型使用）- 非流式
"""
import pytest_check as check
from ...utility import request_helper as helper
from ...utility import assertion


def test_chat_template_kwargs_enable_thinking_true(api_client, request):
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
        "stream": False,
        "max_tokens": 5120
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)


    # 校验点2：finish_reason为stop或length
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)

    # 校验点3：reasoning_content或reasoning非空，content在finish_reason=length时允许为空
    content = response.json()["choices"][0]["message"]["content"] or ""
    reasoning_content = response.json()["choices"][0]["message"].get("reasoning_content", "") or response.json()["choices"][0]["message"].get("reasoning", "") or ""
    check.is_true(len(reasoning_content) > 0, "reasoning_content/reasoning为空")
    check.is_true(not reasoning_content[0].isspace(), "reasoning_content首字符为空字符")
    if finish_reason != "length":
        check.is_true(len(content) > 0, "返回内容为空")
        check.is_true(not content[0].isspace(), "返回内容首字符为空字符")


def test_chat_template_kwargs_enable_thinking_false(api_client, request):
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
        "stream": False,
        "max_tokens": 5120
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)


    # 校验点2：finish_reason为stop或length
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)

    # 校验点3：content非空，reasoning_content和reasoning为空
    content = response.json()["choices"][0]["message"]["content"] or ""
    reasoning_content = response.json()["choices"][0]["message"].get("reasoning_content", "") or response.json()["choices"][0]["message"].get("reasoning", "") or ""
    check.is_true(len(content) > 0, "返回内容为空")
    check.is_true(not content[0].isspace(), "返回内容首字符为空字符")
    check.is_true(len(reasoning_content) == 0, "enable_thinking=false时reasoning_content/reasoning应为空")


# ==================== 异常场景 ====================

def test_chat_template_kwargs_enable_thinking_null(api_client, request):
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
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)


def test_chat_template_kwargs_enable_thinking_string(api_client, request):
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
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    