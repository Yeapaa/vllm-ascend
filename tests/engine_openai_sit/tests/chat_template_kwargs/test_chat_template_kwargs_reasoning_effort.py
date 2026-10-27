import json

import allure
import pytest_check as check

from ...utility import assertion
from ...utility import request_helper as helper


REASONING_EFFORTS = [
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max"
]


def _get_stream_usage(response_text):
    """从流式响应末尾的usage chunk中获取usage。"""
    usage = {}
    for line in response_text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            continue
        chunk = json.loads(data)
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
    return usage


def _validate_and_record_usage(usage, reasoning_effort, usage_records):
    """校验并记录当前reasoning_effort对应的token usage。"""
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")

    check.is_true(
        type(prompt_tokens) is int and prompt_tokens >= 0,
        f"reasoning_effort={reasoning_effort}时prompt_tokens应为非负整数，实际为{prompt_tokens}"
    )
    check.is_true(
        type(completion_tokens) is int and completion_tokens >= 0,
        f"reasoning_effort={reasoning_effort}时completion_tokens应为非负整数，实际为{completion_tokens}"
    )

    usage_record = {
        "reasoning_effort": reasoning_effort,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens
    }
    usage_records.append(usage_record)
    allure.attach(
        json.dumps(usage_record, indent=2, ensure_ascii=False),
        name=f"reasoning_effort={reasoning_effort} usage",
        attachment_type=allure.attachment_type.JSON,
        extension="json"
    )


def _run_reasoning_effort_requests(api_client, stream):
    """按固定顺序请求不同reasoning_effort并汇总usage。"""
    usage_records = []

    for reasoning_effort in REASONING_EFFORTS:
        request_body = {
            "model": "auto",
            "messages": [{
                "role": "user",
                "content": "请逐步推理并回答：若一个数的3倍加5等于26，这个数是多少？"
            }],
            "chat_template_kwargs": {
                "thinking": True,
                "enable_thinking": True,
                "reasoning_effort": reasoning_effort
            },
            "stream": stream,
            "max_tokens": 20480
        }
        if stream:
            request_body["stream_options"] = {
                "include_usage": True
            }

        helper.attach_request_body(
            request_body,
            name=f"reasoning_effort={reasoning_effort} 请求体"
        )
        response = helper.send_request(api_client, "/v1/chat/completions", request_body)
        helper.attach_response_body(
            response,
            name=f"reasoning_effort={reasoning_effort} 响应体"
        )

        if stream:
            assertion.assert_stream_has_done(
                response.text,
                msg=f"reasoning_effort={reasoning_effort}："
            )
            usage = _get_stream_usage(response.text)
        else:
            usage = response.json().get("usage", {})

        _validate_and_record_usage(usage, reasoning_effort, usage_records)

    allure.attach(
        json.dumps(usage_records, indent=2, ensure_ascii=False),
        name="reasoning_effort usage汇总",
        attachment_type=allure.attachment_type.JSON,
        extension="json"
    )


def test_chat_template_kwargs_reasoning_effort_non_stream(api_client):
    """非流式顺序请求各reasoning_effort并记录token usage。"""
    _run_reasoning_effort_requests(api_client, stream=False)


def test_chat_template_kwargs_reasoning_effort_stream(api_client):
    """流式顺序请求各reasoning_effort并记录token usage。"""
    _run_reasoning_effort_requests(api_client, stream=True)
