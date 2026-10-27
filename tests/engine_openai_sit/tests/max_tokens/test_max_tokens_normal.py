import json

import pytest
import pytest_check as check
from ...utility import request_helper as helper
from ...utility import assertion


def _get_reasoning_tokens(usage, engine_type):
    """根据引擎类型从usage中获取reasoning_tokens。"""
    if "sglang" in engine_type.lower():
        return usage.get("reasoning_tokens")
    completion_tokens_details = usage.get("completion_tokens_details", {})
    return completion_tokens_details.get("reasoning_tokens")


def _get_stream_reasoning_usage(response, engine_type):
    """获取流式响应中reasoning_content字段是否存在及reasoning_tokens。"""
    usage = {}
    has_reasoning_content = False
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            continue
        chunk = json.loads(data)
        choices = chunk.get("choices", [])
        if not has_reasoning_content and choices and "reasoning_content" in choices[0].get("delta", {}):
            has_reasoning_content = True
        if chunk.get("usage"):
            usage = chunk["usage"]

    reasoning_tokens = _get_reasoning_tokens(usage, engine_type)
    return has_reasoning_content, reasoning_tokens


@pytest.mark.parametrize("max_tokens", [10, 10000], ids=["10", "10000"])
def test_usage_reasoning_tokens_non_stream(api_client, request, max_tokens):
    """非流式响应的reasoning_tokens为正整数。"""
    engine_architecture = request.config.getoption("--engineArchitecture", default="pd").strip()
    if engine_architecture != "single":
        pytest.skip("reasoning_tokens校验仅支持single单机架构")
    engine_type = request.config.getoption("--engineType", default="vllm").strip()

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请逐步推理并回答：若3个苹果售价12元，购买7个苹果需要多少钱？"
        }],
        "max_tokens": max_tokens,
        "stream": False
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    response_json = response.json()
    choices = response_json.get("choices", [])
    message = choices[0].get("message", {}) if choices else {}
    if "reasoning_content" in message:
        reasoning_tokens = _get_reasoning_tokens(response_json.get("usage", {}), engine_type)
        check.is_instance(reasoning_tokens, int, "reasoning_tokens不存在或不是整数")
        if isinstance(reasoning_tokens, int):
            check.greater(reasoning_tokens, 0, "reasoning_tokens应大于0")


@pytest.mark.parametrize("max_tokens", [10, 10000], ids=["10", "10000"])
def test_usage_reasoning_tokens_stream(api_client, request, max_tokens):
    """流式响应的reasoning_tokens为正整数。"""
    engine_architecture = request.config.getoption("--engineArchitecture", default="pd").strip()
    if engine_architecture != "single":
        pytest.skip("reasoning_tokens校验仅支持single单机架构")
    engine_type = request.config.getoption("--engineType", default="vllm").strip()

    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请逐步推理并回答：若3个苹果售价12元，购买7个苹果需要多少钱？"
        }],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {
            "include_usage": True
        }
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_stream_has_done(response.text)
    has_reasoning_content, reasoning_tokens = _get_stream_reasoning_usage(response, engine_type)
    if has_reasoning_content:
        check.is_instance(reasoning_tokens, int, "reasoning_tokens不存在或不是整数")
        if isinstance(reasoning_tokens, int):
            check.greater(reasoning_tokens, 0, "reasoning_tokens应大于0")


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("max_tokens", [1, 10, 50, 100, 512, 2048, 4096],
                         ids=["1", "10", "50", "100", "512", "2048", "4096"])
def test_max_tokens_normal_values(api_client, request, stream, max_tokens):
    """max_tokens正常取值范围[1, 模型上限]，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请用一句话介绍自己。"
        }],
        "max_tokens": max_tokens,
        "stream": stream
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)


    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason有效
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_boundary_1(api_client, request, stream):
    """max_tokens=1边界值，只生成1个token"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_tokens": 1,
        "stream": stream
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)


    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason通常为length
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_large_value(api_client, request, stream):
    """max_tokens较大值（如8192），请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "写一个简短的故事"
        }],
        "max_tokens": 8192,
        "stream": stream
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)


    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason有效
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_without_setting(api_client, request, stream):
    """请求体中不设置max_tokens参数，使用默认值，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        # 不设置max_tokens
        "stream": stream
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)


    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason有效
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_with_temperature(api_client, request, stream):
    """max_tokens与temperature组合使用，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请写一段描述"
        }],
        "max_tokens": 100,
        "temperature": 0.8,
        "stream": stream
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)


    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason有效
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_with_stop(api_client, request, stream):
    """max_tokens与stop组合使用，先到限制者生效"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请写一段描述，包含结束标记"
        }],
        "max_tokens": 200,
        "stop": ["结束"],
        "stream": stream
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)


    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason有效
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_with_top_p(api_client, request, stream):
    """max_tokens与top_p组合使用，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请写一个短句"
        }],
        "max_tokens": 50,
        "top_p": 0.9,
        "stream": stream
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)


    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_with_presence_penalty(api_client, request, stream):
    """max_tokens与presence_penalty组合使用，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "请列举一些动物"
        }],
        "max_tokens": 100,
        "presence_penalty": 0.5,
        "stream": stream
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)


    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_max_tokens_null(api_client, request, stream):
    """max_tokens为null，使用默认值，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "max_tokens": None,
        "stream": stream
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)


    # 校验点2：流式响应包含[DONE]
    if stream:
        assertion.assert_stream_has_done(response.text)

    # 校验点3：finish_reason有效
    if stream:
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    else:
        finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)
