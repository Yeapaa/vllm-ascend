"""Kimi preserve_thinking字段测试。

从Kimi K2.6开始支持该feature，Kimi K2.7-Code/K3强制开启。
"""

import pytest

from ...utility import assertion
from ...utility import request_helper as helper


PROBE_CODE = "RC_ONLY_PROBE_20260707_X9Q2"
PRESERVE_THINKING_OMITTED = object()

MESSAGES = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "请计算 17 * 23，并简单说明。"
            }
        ]
    },
    {
        "role": "assistant",
        "content": "17 * 23 = 391。简单说明：17 * (20 + 3) = 340 + 51 = 391。",
        "reasoning_content": (
            "我在这里记录一个只存在于 reasoning_content 的探测码："
            "RC_ONLY_PROBE_20260707_X9Q2。当前回答不能提到这个探测码，"
            "如果用户下一轮问，可以回答。"
        )
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "你上一轮思考过程里记录的探测码是什么？只回答探测码。"
            }
        ]
    }
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer"
                    },
                    "b": {
                        "type": "integer"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
]


def _build_request_body(preserve_thinking, with_tools):
    """构造preserve_thinking测试请求体，支持字段缺省场景。"""
    chat_template_kwargs = {
        "enable_thinking": True,
        "thinking": True
    }
    if preserve_thinking is not PRESERVE_THINKING_OMITTED:
        chat_template_kwargs["preserve_thinking"] = preserve_thinking

    request_body = {
        "messages": MESSAGES,
        "model": "auto",
        "chat_template_kwargs": chat_template_kwargs
    }
    if with_tools:
        request_body["tools"] = TOOLS
    return request_body


def _send_request_and_get_content(api_client, request_body):
    """发送请求并返回choices[0].message.content。"""
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    response_json = response.json()
    finish_reason = response_json["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)

    content = response_json["choices"][0]["message"].get("content", "") or ""
    assert isinstance(content, str), f"响应content应为字符串，实际类型为{type(content).__name__}"
    return content


@pytest.mark.parametrize(
    "preserve_thinking, expected_contains_probe",
    [
        (False, False),
        (True, True),
        (PRESERVE_THINKING_OMITTED, False)
    ],
    ids=["preserve_false", "preserve_true", "preserve_omitted"]
)
def test_kimi_preserve_thinking_without_tools(
    api_client,
    preserve_thinking,
    expected_contains_probe
):
    """无tools时，preserve_thinking控制历史reasoning_content是否进入上下文。"""
    request_body = _build_request_body(preserve_thinking, with_tools=False)
    content = _send_request_and_get_content(api_client, request_body)

    if expected_contains_probe:
        assert PROBE_CODE in content, f"响应content中应包含探测码{PROBE_CODE}"
    else:
        assert PROBE_CODE not in content, f"响应content中不应包含探测码{PROBE_CODE}"


@pytest.mark.parametrize(
    "preserve_thinking, expected_contains_probe",
    [
        (False, False),
        (True, True),
        (PRESERVE_THINKING_OMITTED, False)
    ],
    ids=["preserve_false", "preserve_true", "preserve_omitted"]
)
def test_kimi_preserve_thinking_with_tools(
    api_client,
    preserve_thinking,
    expected_contains_probe
):
    """有tools时，preserve_thinking同样控制历史reasoning_content是否进入上下文。"""
    request_body = _build_request_body(preserve_thinking, with_tools=True)
    content = _send_request_and_get_content(api_client, request_body)

    if expected_contains_probe:
        assert PROBE_CODE in content, f"响应content中应包含探测码{PROBE_CODE}"
    else:
        assert PROBE_CODE not in content, f"响应content中不应包含探测码{PROBE_CODE}"
