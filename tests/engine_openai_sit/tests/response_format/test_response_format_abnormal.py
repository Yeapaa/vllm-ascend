import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_missing_json_schema_field(api_client, request, stream):
    """response_format type=json_schema但缺少json_schema字段，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {"type": "json_schema"},
        "stream": stream,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_invalid_type(api_client, request, stream):
    """response_format type为无效值（如xml），应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {"type": "xml"},
        "stream": stream,
        "max_tokens": 50
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_invalid_type_html(api_client, request, stream):
    """response_format type为无效值（如html），应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {"type": "html"},
        "stream": stream,
        "max_tokens": 50
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_invalid_type_yaml(api_client, request, stream):
    """response_format type为无效值（如yaml），应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {"type": "yaml"},
        "stream": stream,
        "max_tokens": 50
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_empty_object(api_client, request, stream):
    """response_format为空对象{}，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {},
        "stream": stream,
        "max_tokens": 50
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_string_type(api_client, request, stream):
    """response_format为字符串类型（非对象），应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": "json_object",
        "stream": stream,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_array_type(api_client, request, stream):
    """response_format为数组类型，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": [{"type": "json_object"}],
        "stream": stream,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_empty_json_schema(api_client, request, stream):
    """response_format type=json_schema但json_schema为空对象，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {}
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
def test_response_format_json_schema_missing_name(api_client, request, stream):
    """response_format type=json_schema但json_schema中缺少name字段，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
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
def test_response_format_json_schema_missing_schema(api_client, request, stream):
    """response_format type=json_schema但json_schema中缺少schema字段，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema"
            }
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
def test_response_format_json_schema_invalid_schema_type(api_client, request, stream):
    """response_format type=json_schema但schema中type为无效值，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "invalid_type_schema",
                "schema": {
                    "type": "not_a_valid_type"
                }
            }
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
def test_response_format_type_integer(api_client, request, stream):
    """response_format type为整数类型，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {"type": 123},
        "stream": stream,
        "max_tokens": 50
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_type_null(api_client, request, stream):
    """response_format type为null，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {"type": None},
        "stream": stream,
        "max_tokens": 50
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_name_is_number(api_client, request, stream):
    """response_format type=json_schema但name为数字类型，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": 12345,
                "schema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"}
                    }
                }
            }
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
def test_response_format_json_schema_schema_is_string(api_client, request, stream):
    """response_format type=json_schema但schema为字符串类型，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": "not an object"
            }
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
def test_response_format_json_schema_schema_is_array(api_client, request, stream):
    """response_format type=json_schema但schema为数组类型，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": [{"type": "string"}]
            }
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
def test_response_format_json_schema_with_extra_unknown_fields(api_client, request, stream):
    """response_format包含未知额外字段，应返回非5xx错误或正常忽略"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_object",
            "unknown_field": "some_value",
            "another_extra": 42
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
def test_response_format_json_object_with_tool_choice(api_client, request, stream):
    """response_format type=json_object同时指定tool_choice，可能冲突，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {"type": "json_object"},
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                }
            }
        }],
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
def test_response_format_json_schema_with_named_tool_choice(api_client, request, stream):
    """response_format type=json_schema同时指定具名tool_choice，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string"}
                    }
                }
            }
        },
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                }
            }
        }],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "stream": stream,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_response_format_json_schema_strict_not_bool_non_stream(api_client, request):
    """非流式：json_schema.strict为非布尔类型，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "strict": "yes",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False
                }
            }
        },
        "stream": False,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_response_format_json_schema_strict_not_bool_stream(api_client, request):
    """流式：json_schema.strict为非布尔类型，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "strict": "yes",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False
                }
            }
        },
        "stream": True,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_response_format_json_schema_with_structured_outputs_json_non_stream(api_client, request):
    """非流式：同时使用response_format json_schema和structured_outputs.json，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False
                }
            }
        },
        "structured_outputs": {
            "json": {"type": "object", "properties": {"value": {"type": "integer"}}}
        },
        "stream": False,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_response_format_json_schema_with_structured_outputs_json_stream(api_client, request):
    """流式：同时使用response_format json_schema和structured_outputs.json，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False
                }
            }
        },
        "structured_outputs": {
            "json": {"type": "object", "properties": {"value": {"type": "integer"}}}
        },
        "stream": True,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_response_format_json_schema_invalid_schema_type_non_stream(api_client, request):
    """非流式：json_schema.schema的type不是object，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "strict": True,
                "schema": {"type": "string"}
            }
        },
        "stream": False,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_response_format_json_schema_invalid_schema_type_stream(api_client, request):
    """流式：json_schema.schema的type不是object，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "strict": True,
                "schema": {"type": "string"}
            }
        },
        "stream": True,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_response_format_json_schema_empty_schema_non_stream(api_client, request):
    """非流式：json_schema.schema为空对象{}，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "strict": True,
                "schema": {}
            }
        },
        "stream": False,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


def test_response_format_json_schema_empty_schema_stream(api_client, request):
    """流式：json_schema.schema为空对象{}，应返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "strict": True,
                "schema": {}
            }
        },
        "stream": True,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)
