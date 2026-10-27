import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_text(api_client, request, stream):
    """response_format type=text，纯文本输出，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "用一句话形容夏天的夜晚。"}],
        "response_format": {"type": "text"},
        "stream": stream,
        "max_tokens": 50
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_object(api_client, request, stream):
    """response_format type=json_object，JSON对象输出，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请以JSON格式返回一个包含name和age字段的人物信息。"}
        ],
        "response_format": {"type": "json_object"},
        "stream": stream,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.validate_json_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_simple(api_client, request, stream):
    """response_format type=json_schema，简单JSON Schema，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个包含name和age字段的人物信息。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person_info",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"]
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
    assertion.validate_json_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_strict_true(api_client, request, stream):
    """response_format type=json_schema，strict=true，严格模式，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个包含name和age字段的人物信息。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person_info",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False
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
    assertion.validate_json_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_with_description(api_client, request, stream):
    """response_format type=json_schema，带description字段，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个人物信息。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person_info",
                "description": "A simple person information schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"]
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
    assertion.validate_json_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_nested_object(api_client, request, stream):
    """response_format type=json_schema，嵌套对象Schema，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个包含地址信息的人物信息。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person_with_address",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "street": {"type": "string"},
                                "zip_code": {"type": "string"}
                            },
                            "required": ["city", "street"]
                        }
                    },
                    "required": ["name", "address"]
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
    assertion.validate_json_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_array(api_client, request, stream):
    """response_format type=json_schema，数组类型Schema，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个包含多个水果名称的列表。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "fruit_list",
                "schema": {
                    "type": "object",
                    "properties": {
                        "fruits": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["fruits"]
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
    assertion.validate_json_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_enum(api_client, request, stream):
    """response_format type=json_schema，包含enum枚举值的Schema，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请返回一个颜色信息。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "color_info",
                "schema": {
                    "type": "object",
                    "properties": {
                        "color": {
                            "type": "string",
                            "enum": ["red", "green", "blue", "yellow"]
                        },
                        "hex_code": {"type": "string"}
                    },
                    "required": ["color"]
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
    assertion.validate_json_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_all_basic_types(api_client, request, stream):
    """response_format type=json_schema，包含所有基本类型的Schema，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个包含多种类型字段的数据。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "multi_type_data",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "score": {"type": "number"},
                        "is_active": {"type": "boolean"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["name", "age"]
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
    assertion.validate_json_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_strict_false(api_client, request, stream):
    """response_format type=json_schema，strict=false，非严格模式，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个人物信息。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person_info",
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"],
                    "additionalProperties": True
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
    assertion.validate_json_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_optional_fields(api_client, request, stream):
    """response_format type=json_schema，含可选字段的Schema，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个用户信息。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "user_info",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "phone": {"type": "string"}
                    },
                    "required": ["name"]
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
    assertion.validate_json_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_object_with_json_prompt(api_client, request, stream):
    """response_format type=json_object，prompt中明确要求JSON输出，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": "请始终以JSON格式回复。"},
            {"role": "user", "content": "返回一个包含城市和温度的天气信息。"}
        ],
        "response_format": {"type": "json_object"},
        "stream": stream,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.validate_json_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_string_format(api_client, request, stream):
    """response_format type=json_schema，Schema中string类型使用format属性，请求正常处理"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个用户信息，包含邮箱和日期。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "user_with_format",
                "schema": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "format": "email"},
                        "birthday": {"type": "string", "format": "date"}
                    },
                    "required": ["email", "birthday"]
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
    assertion.validate_json_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_complex(api_client, request, stream):
    """复杂嵌套schema（含数组、嵌套对象），正常生成结构化输出"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "描述一个人的信息，包括姓名、年龄、爱好和地址"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person_detail",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "hobbies": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "address": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "zip": {"type": "string"}
                            },
                            "required": ["city"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["name"],
                    "additionalProperties": False
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
    assertion.validate_json_response(response, stream)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_with_temperature(api_client, request, stream):
    """response_format为json_schema并设置temperature，正常生成结构化输出"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个人物信息"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False
                }
            }
        },
        "temperature": 0.0,
        "stream": stream,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.validate_json_response(response, stream)
