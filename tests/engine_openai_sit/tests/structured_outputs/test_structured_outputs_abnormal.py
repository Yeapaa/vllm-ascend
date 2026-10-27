import pytest
import json
from ...utility import request_helper as helper
from ...utility import assertion


# ============================================================
# structured_outputs 多约束冲突异常用例
# ============================================================


def test_structured_outputs_json_and_regex_conflict_non_stream(api_client, request):
    """非流式：structured_outputs同时指定json和regex，应返回错误（只能指定一种约束）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "json": {"type": "object", "properties": {"name": {"type": "string"}}},
            "regex": r"\d+"
        },
        "stream": False,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_json_and_regex_conflict_stream(api_client, request):
    """流式：structured_outputs同时指定json和regex，应返回错误（只能指定一种约束）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "json": {"type": "object", "properties": {"name": {"type": "string"}}},
            "regex": r"\d+"
        },
        "stream": True,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_json_and_choice_conflict_non_stream(api_client, request):
    """非流式：structured_outputs同时指定json和choice，应返回错误（只能指定一种约束）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "json": {"type": "object", "properties": {"name": {"type": "string"}}},
            "choice": ["A", "B"]
        },
        "stream": False,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_json_and_choice_conflict_stream(api_client, request):
    """流式：structured_outputs同时指定json和choice，应返回错误（只能指定一种约束）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "json": {"type": "object", "properties": {"name": {"type": "string"}}},
            "choice": ["A", "B"]
        },
        "stream": True,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_regex_and_choice_conflict_non_stream(api_client, request):
    """非流式：structured_outputs同时指定regex和choice，应返回错误（只能指定一种约束）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "regex": r"\d+",
            "choice": ["A", "B"]
        },
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_regex_and_choice_conflict_stream(api_client, request):
    """流式：structured_outputs同时指定regex和choice，应返回错误（只能指定一种约束）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "regex": r"\d+",
            "choice": ["A", "B"]
        },
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_json_and_grammar_conflict_non_stream(api_client, request):
    """非流式：structured_outputs同时指定json和grammar，应返回错误（只能指定一种约束）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "json": {"type": "object", "properties": {"name": {"type": "string"}}},
            "grammar": 'root ::= "yes" | "no"'
        },
        "stream": False,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_json_and_grammar_conflict_stream(api_client, request):
    """流式：structured_outputs同时指定json和grammar，应返回错误（只能指定一种约束）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "json": {"type": "object", "properties": {"name": {"type": "string"}}},
            "grammar": 'root ::= "yes" | "no"'
        },
        "stream": True,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


# ============================================================
# structured_outputs 字段类型错误异常用例
# ============================================================


def test_structured_outputs_json_invalid_type_non_stream(api_client, request):
    """非流式：structured_outputs.json为整数类型，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "json": 123
        },
        "stream": False,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_json_invalid_type_stream(api_client, request):
    """流式：structured_outputs.json为整数类型，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "json": 123
        },
        "stream": True,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_regex_invalid_type_non_stream(api_client, request):
    """非流式：structured_outputs.regex为非字符串类型，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "regex": 12345
        },
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_regex_invalid_type_stream(api_client, request):
    """流式：structured_outputs.regex为非字符串类型，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "regex": 12345
        },
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_choice_invalid_type_non_stream(api_client, request):
    """非流式：structured_outputs.choice为非数组类型，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "choice": "single_value"
        },
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_choice_invalid_type_stream(api_client, request):
    """流式：structured_outputs.choice为非数组类型，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "choice": "single_value"
        },
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


# ============================================================
# structured_outputs 与 tools 冲突异常用例
# ============================================================


def test_structured_outputs_json_with_tool_choice_named_non_stream(api_client, request):
    """非流式：structured_outputs与命名tool_choice同时使用，应返回错误"""
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "北京天气怎么样"}],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "structured_outputs": {
            "json": {"type": "object", "properties": {"result": {"type": "string"}}}
        },
        "stream": False,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_json_with_tool_choice_named_stream(api_client, request):
    """流式：structured_outputs与命名tool_choice同时使用，应返回错误"""
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]

    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "北京天气怎么样"}],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "structured_outputs": {
            "json": {"type": "object", "properties": {"result": {"type": "string"}}}
        },
        "stream": True,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


# ============================================================
# structured_outputs 无效正则表达式异常用例
# ============================================================


def test_structured_outputs_invalid_regex_non_stream(api_client, request):
    """非流式：structured_outputs.regex为无效正则表达式，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "regex": "[invalid(regex"
        },
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_invalid_regex_stream(api_client, request):
    """流式：structured_outputs.regex为无效正则表达式，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "regex": "[invalid(regex"
        },
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


# ============================================================
# structured_outputs choice 空列表异常用例
# ============================================================


def test_structured_outputs_choice_empty_list_non_stream(api_client, request):
    """非流式：structured_outputs.choice为空列表，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "choice": []
        },
        "stream": False,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_structured_outputs_choice_empty_list_stream(api_client, request):
    """流式：structured_outputs.choice为空列表，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "choice": []
        },
        "stream": True,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)
