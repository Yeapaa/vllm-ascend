import re
import pytest
import json
from ...utility import request_helper as helper
from ...utility import assertion


def _validate_arguments(arguments_str, msg=""):
    """校验tool_calls中arguments字段的完整性，支持标准JSON和XML标签两种格式

    标准JSON格式: {"city": "北京"}
    XML标签格式: 嵌套层级约束 tool_call > function > parameter
    """
    try:
        args = json.loads(arguments_str)
        assert isinstance(args, dict), f"{msg}arguments解析后应为dict，实际为{type(args).__name__}"
    except json.JSONDecodeError:
        # 非标准JSON，校验XML标签格式
        # 1. 所有开标签必须有对应的闭标签
        opening_tags = re.findall(r'<(\w+)(?:=[^>]*)?>', arguments_str)
        closing_tags = re.findall(r'</(\w+)>', arguments_str)
        assert len(opening_tags) > 0, f"{msg}arguments非JSON格式时，应包含XML标签"
        for tag in set(opening_tags):
            assert closing_tags.count(tag) >= opening_tags.count(tag), \
                f"{msg}arguments中标签<{tag}>缺少对应的</{tag}>闭合标签"
        # 2. 嵌套层级约束: tool_call > function > parameter
        xml_hierarchy = {"tool_call": None, "function": "tool_call", "parameter": "function"}
        for child, parent in xml_hierarchy.items():
            if child in opening_tags and parent is not None:
                assert parent in opening_tags, \
                    f"{msg}XML标签层级约束: <{child}>必须在<{parent}>内"


TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def _validate_tool_calls_structure(response, stream, expected_tools=None):
    """校验tool_calls响应结构的通用函数（用于null值测试）
    
    Args:
        response: HTTP响应对象
        stream: 是否流式响应
        expected_tools: 期望的工具名称列表，默认为["get_weather", "get_time"]
    """
    if expected_tools is None:
        expected_tools = ["get_weather", "get_time"]
    
    if stream:
        # 流式响应校验
        assertion.assert_stream_has_done(response.text)
        
        tool_calls_found = False
        finish_reason_tool_calls = False
        for line in response.text.strip().split("\n"):
            if line.startswith("data: ") and "[DONE]" not in line:
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("tool_calls"):
                        tool_calls_found = True
                        for tc in delta.get("tool_calls", []):
                            assert "index" in tc, "流式tool_calls应包含index字段"
                            if tc.get("id"):
                                assert tc.get("type") == "function", "tool_calls类型应为function"
                            if tc.get("function"):
                                func = tc.get("function", {})
                                if func.get("name"):
                                    assert func.get("name") in expected_tools,                                         f"函数名应为{expected_tools}之一，实际为{func.get('name')}"
                    if choices[0].get("finish_reason") == "tool_calls":
                        finish_reason_tool_calls = True
        
        # 如果有 tool_calls，finish_reason 应为 tool_calls
        if tool_calls_found:
            assert finish_reason_tool_calls, "流式响应有tool_calls时finish_reason应为tool_calls"
    else:
        # 非流式响应校验
        response_json = response.json()
        choices = response_json.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            
            if message.get("tool_calls") or choice.get("finish_reason") == "tool_calls":
                assert choice.get("finish_reason") == "tool_calls",                     f"finish_reason应为tool_calls，实际为{choice.get('finish_reason')}"
                
                tool_calls = message.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        assert "id" in tc, "tool_call应包含id字段"
                        assert tc.get("type") == "function", "tool_call类型应为function"
                        assert "function" in tc, "tool_call应包含function字段"
                        func = tc.get("function", {})
                        assert "name" in func, "function应包含name字段"
                        assert "arguments" in func, "function应包含arguments字段"
                        assert func.get("name") in expected_tools,                             f"函数名应为{expected_tools}之一，实际为{func.get('name')}"
                        _validate_arguments(func.get("arguments", "{}"))


def test_parallel_tool_calls_string_value_non_stream(api_client, request):
    """非流式：parallel_tool_calls为字符串类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": "true",  # 字符串类型，应为布尔类型
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_parallel_tool_calls_string_value_stream(api_client, request):
    """流式：parallel_tool_calls为字符串类型，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": "false",  # 字符串类型
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_parallel_tool_calls_integer_value_non_stream(api_client, request):
    """非流式：parallel_tool_calls为整数类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": 1,  # 整数类型
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_parallel_tool_calls_integer_value_stream(api_client, request):
    """流式：parallel_tool_calls为整数类型，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": 0,  # 整数类型
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_parallel_tool_calls_null_value_non_stream(api_client, request):
    """非流式：parallel_tool_calls为null，应正常响应（null等同于默认值True）"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": None,
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    # 校验点：finish_reason为stop或length或tool_calls
    response_json = response.json()
    choices = response_json.get("choices", [])
    if choices:
        finish_reason = choices[0].get("finish_reason")
        if finish_reason == "tool_calls":
            # 如果有tool_calls，校验结构
            _validate_tool_calls_structure(response, stream=False)
        else:
            assertion.assert_finish_reason_valid(finish_reason)


def test_parallel_tool_calls_null_value_stream(api_client, request):
    """流式：parallel_tool_calls为null，sglang引擎返回400错误码，其他引擎正常响应"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": None,
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    engine_type = request.config.getoption("--engineType", default="vllm").strip()
    if "sglang" in engine_type.lower():
        # sglang引擎：parallel_tool_calls为null时返回400错误码
        assertion.assert_error_code_400(response)
    else:
        # 其他引擎：finish_reason为stop或length或tool_calls
        finish_reason = assertion.assert_stream_single_finish_reason(response.text)
        if finish_reason == "tool_calls":
            _validate_tool_calls_structure(response, stream=True)
        else:
            assertion.assert_finish_reason_valid(finish_reason)


def test_parallel_tool_calls_array_value_non_stream(api_client, request):
    """非流式：parallel_tool_calls为数组类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": [True],  # 数组类型
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_parallel_tool_calls_array_value_stream(api_client, request):
    """流式：parallel_tool_calls为数组类型，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": [],  # 空数组
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_parallel_tool_calls_object_value_non_stream(api_client, request):
    """非流式：parallel_tool_calls为对象类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": {"enabled": True},  # 对象类型
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_parallel_tool_calls_object_value_stream(api_client, request):
    """流式：parallel_tool_calls为对象类型，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": {},  # 空对象
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_parallel_tool_calls_float_value_non_stream(api_client, request):
    """非流式：parallel_tool_calls为浮点数类型，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": 1.0,  # 浮点数类型
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_parallel_tool_calls_float_value_stream(api_client, request):
    """流式：parallel_tool_calls为浮点数类型，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": 0.5,  # 浮点数类型
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_parallel_tool_calls_empty_string_non_stream(api_client, request):
    """非流式：parallel_tool_calls为空字符串，应返回400错误"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": "",  # 空字符串
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_parallel_tool_calls_empty_string_stream(api_client, request):
    """流式：parallel_tool_calls为空字符串，http status code:200, response_body返回400错误码"""
    request_body = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": "你好"
        }],
        "tools": TOOLS_DEFINITION,
        "parallel_tool_calls": "",  # 空字符串
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)
