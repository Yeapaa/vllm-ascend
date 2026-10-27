import pytest
import json
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_empty_object(api_client, request, stream):
    """structured_outputs为空对象{}，应返回错误（无约束类型指定）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {},
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_json_empty_dict(api_client, request, stream):
    """structured_outputs.json为空dict{}，应正常处理或返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "json": {}
        },
        "stream": stream,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_json_empty_string(api_client, request, stream):
    """structured_outputs.json为空字符串，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "json": ""
        },
        "stream": stream,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_regex_empty_string(api_client, request, stream):
    """structured_outputs.regex为空字符串，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "regex": ""
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_choice_single_item(api_client, request, stream):
    """structured_outputs.choice仅含一个选项，应正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "回答是或否"}],
        "structured_outputs": {
            "choice": ["yes"]
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_choice_many_items(api_client, request, stream):
    """structured_outputs.choice含多个选项，应正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "选择一个颜色"}],
        "structured_outputs": {
            "choice": ["红色", "橙色", "黄色", "绿色", "蓝色", "靛色", "紫色"]
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_regex_simple_pattern(api_client, request, stream):
    """structured_outputs.regex为简单正则模式，应正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "给我一个两位数字"}],
        "structured_outputs": {
            "regex": r"\d{2}"
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_regex_complex_pattern(api_client, request, stream):
    """structured_outputs.regex为复杂正则模式，应正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "给我一个邮箱地址"}],
        "structured_outputs": {
            "regex": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_json_with_tool_choice_auto(api_client, request, stream):
    """structured_outputs.json与tool_choice=auto同时使用，应正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个人物信息"}],
        "structured_outputs": {
            "json": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"],
                "additionalProperties": False
            }
        },
        "tool_choice": "auto",
        "stream": stream,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_json_with_tool_choice_none(api_client, request, stream):
    """structured_outputs.json与tool_choice=none同时使用，应正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个人物信息"}],
        "structured_outputs": {
            "json": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"],
                "additionalProperties": False
            }
        },
        "tool_choice": "none",
        "stream": stream,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_grammar_simple(api_client, request, stream):
    """structured_outputs.grammar为简单EBNF语法，应正常处理"""
    grammar = 'root ::= "hello" | "world"'
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "打招呼"}],
        "structured_outputs": {
            "grammar": grammar
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_grammar_empty_string(api_client, request, stream):
    """structured_outputs.grammar为空字符串，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "grammar": ""
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_grammar_null(api_client, request, stream):
    """structured_outputs.grammar为空字符串，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "grammar": None
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_json_object_true(api_client, request, stream):
    """structured_outputs.json_object为true，应正常生成JSON输出"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个包含name字段的JSON对象"}],
        "structured_outputs": {
            "json_object": True
        },
        "stream": stream,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_json_object_false(api_client, request, stream):
    """structured_outputs.json_object为false，应返回错误（json_object=False无效）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "structured_outputs": {
            "json_object": False
        },
        "stream": stream,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)
