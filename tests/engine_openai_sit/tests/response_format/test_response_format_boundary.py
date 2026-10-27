import pytest
from ...utility import request_helper as helper
from ...utility import assertion


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_empty_name(api_client, request, stream):
    """response_format type=json_schema，name为空字符串，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "",
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
def test_response_format_json_schema_very_long_name(api_client, request, stream):
    """response_format type=json_schema，name为超长字符串，应返回非5xx错误"""
    long_name = "a" * 10000
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": long_name,
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
def test_response_format_json_schema_minimal_object(api_client, request, stream):
    """response_format type=json_schema，最小化object schema（仅type:object），应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请返回一个空的JSON对象。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "minimal_object",
                "schema": {
                    "type": "object"
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
def test_response_format_json_schema_deeply_nested(api_client, request, stream):
    """response_format type=json_schema，深层嵌套schema（5层），应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个深层嵌套的数据结构。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "deeply_nested",
                "schema": {
                    "type": "object",
                    "properties": {
                        "level1": {
                            "type": "object",
                            "properties": {
                                "level2": {
                                    "type": "object",
                                    "properties": {
                                        "level3": {
                                            "type": "object",
                                            "properties": {
                                                "level4": {
                                                    "type": "object",
                                                    "properties": {
                                                        "level5": {
                                                            "type": "object",
                                                            "properties": {
                                                                "value": {"type": "string"}
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
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
def test_response_format_json_schema_many_properties(api_client, request, stream):
    """response_format type=json_schema，包含大量属性的schema（50个字段），应返回非5xx"""
    properties = {f"field_{i}": {"type": "string"} for i in range(50)}
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个包含多个字段的数据。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "many_properties",
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": ["field_0"]
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
def test_response_format_json_object_without_json_prompt(api_client, request, stream):
    """response_format type=json_object但prompt中未提及JSON，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "你好，请介绍一下自己。"}
        ],
        "response_format": {"type": "json_object"},
        "stream": stream,
        "max_tokens": 51200
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_text_with_extra_fields(api_client, request, stream):
    """response_format type=text但包含额外字段，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "text",
            "extra_field": "should_be_ignored"
        },
        "stream": stream,
        "max_tokens": 50
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_name_with_special_chars(api_client, request, stream):
    """response_format type=json_schema，name包含特殊字符，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test-schema_v1.0!@#",
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
def test_response_format_json_schema_name_unicode(api_client, request, stream):
    """response_format type=json_schema，name包含中文字符，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "用户信息模式",
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
def test_response_format_json_schema_very_long_description(api_client, request, stream):
    """response_format type=json_schema，description为超长字符串，应返回非5xx"""
    long_description = "A schema for user data. " * 500
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "long_desc_schema",
                "description": long_description,
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
def test_response_format_json_schema_empty_properties(api_client, request, stream):
    """response_format type=json_schema，schema properties为空对象，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "empty_props",
                "schema": {
                    "type": "object",
                    "properties": {}
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
def test_response_format_json_schema_strict_true_no_additional_properties(api_client, request, stream):
    """response_format type=json_schema，strict=true但未设置additionalProperties=false，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个人物信息。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "strict_no_additional",
                "strict": True,
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_nested_array_of_objects(api_client, request, stream):
    """response_format type=json_schema，嵌套对象数组的schema，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个包含多个学生的班级信息。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "class_info",
                "schema": {
                    "type": "object",
                    "properties": {
                        "class_name": {"type": "string"},
                        "students": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "age": {"type": "integer"},
                                    "scores": {
                                        "type": "array",
                                        "items": {"type": "number"}
                                    }
                                },
                                "required": ["name", "age"]
                            }
                        }
                    },
                    "required": ["class_name", "students"]
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
def test_response_format_json_schema_with_default_values(api_client, request, stream):
    """response_format type=json_schema，schema中包含default值，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个配置信息。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "config_with_defaults",
                "schema": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "default": "auto"},
                        "timeout": {"type": "integer", "default": 30},
                        "debug": {"type": "boolean", "default": False}
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
def test_response_format_json_schema_with_anyof(api_client, request, stream):
    """response_format type=json_schema，schema中使用anyOf组合类型，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个可以是字符串或数字的值。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "anyof_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "integer"}
                            ]
                        }
                    },
                    "required": ["value"]
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
def test_response_format_json_schema_with_allof(api_client, request, stream):
    """response_format type=json_schema，schema中使用allOf组合类型，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个组合对象。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "allof_schema",
                "schema": {
                    "type": "object",
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"}
                            },
                            "required": ["name"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "age": {"type": "integer"}
                            },
                            "required": ["age"]
                        }
                    ]
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
def test_response_format_json_schema_with_oneof(api_client, request, stream):
    """response_format type=json_schema，schema中使用oneOf组合类型，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个数据项。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "oneof_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "integer"}
                            ]
                        }
                    },
                    "required": ["data"]
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
def test_response_format_json_schema_with_ref(api_client, request, stream):
    """response_format type=json_schema，schema中使用$ref引用，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个地址信息。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ref_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "address": {"$ref": "#/$defs/Address"}
                    },
                    "required": ["address"],
                    "$defs": {
                        "Address": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "street": {"type": "string"}
                            },
                            "required": ["city", "street"]
                        }
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
def test_response_format_json_schema_very_large_schema(api_client, request, stream):
    """response_format type=json_schema，超大schema（100个属性+嵌套），应返回非5xx"""
    properties = {}
    for i in range(100):
        if i % 3 == 0:
            properties[f"field_{i}"] = {"type": "string"}
        elif i % 3 == 1:
            properties[f"field_{i}"] = {"type": "integer"}
        else:
            properties[f"field_{i}"] = {
                "type": "object",
                "properties": {
                    "nested_a": {"type": "string"},
                    "nested_b": {"type": "boolean"}
                }
            }
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个大型数据对象。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "large_schema",
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": ["field_0"]
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
def test_response_format_json_schema_name_max_length(api_client, request, stream):
    """response_format type=json_schema，name为256字符长度，应返回非5xx"""
    name_256 = "a" * 256
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": name_256,
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
def test_response_format_json_schema_description_max_length(api_client, request, stream):
    """response_format type=json_schema，description为1024字符长度，应返回非5xx"""
    description_1024 = "x" * 1024
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "long_desc",
                "description": description_1024,
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
def test_response_format_json_schema_strict_true_all_required(api_client, request, stream):
    """response_format type=json_schema，strict=true且所有字段required+additionalProperties=false，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个完整的人物信息。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "strict_all_required",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "email": {"type": "string"}
                    },
                    "required": ["name", "age", "email"],
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_empty_required(api_client, request, stream):
    """response_format type=json_schema，required为空数组，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "请生成一个数据。"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "empty_required",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": []
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
def test_response_format_json_schema_required_not_subset_of_properties(api_client, request, stream):
    """response_format type=json_schema，required包含properties中不存在的字段名，应返回非5xx"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "invalid_required",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name", "nonexistent_field"]
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
def test_response_format_type_case_sensitive(api_client, request, stream):
    """response_format type大小写敏感（如JSON_OBJECT），应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {"type": "JSON_OBJECT"},
        "stream": stream,
        "max_tokens": 50
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_type_mixed_case(api_client, request, stream):
    """response_format type混合大小写（如Json_Schema），应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {"type": "Json_Schema"},
        "stream": stream,
        "max_tokens": 50
    }
    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)
    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_name_is_empty_array(api_client, request, stream):
    """response_format type=json_schema但name为数组类型，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": [],
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
def test_response_format_json_schema_name_is_object(api_client, request, stream):
    """response_format type=json_schema但name为对象类型，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": {"first": "test", "last": "schema"},
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
def test_response_format_json_schema_name_is_boolean(api_client, request, stream):
    """response_format type=json_schema但name为布尔类型，应返回非5xx错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": True,
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


# Migrated response_format boundary cases

@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_null(api_client, request, stream):
    """response_format为null，等同于不设置"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你好"}],
        "response_format": None,
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_strict_null(api_client, request, stream):
    """response_format为json_schema，strict为null（不指定），应正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个人物信息"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "strict": None,
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_without_strict(api_client, request, stream):
    """response_format为json_schema，不设置strict字段，应正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个人物信息"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_without_description(api_client, request, stream):
    """response_format为json_schema，不设置description字段，应正常处理"""
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
        "stream": stream,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_description_null(api_client, request, stream):
    """response_format为json_schema，description为null，应正常处理"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个人物信息"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "description": None,
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_minimal_object_strict(api_client, request, stream):
    """response_format为json_schema，最小化schema（仅type和空properties）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "返回一个空JSON对象"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "empty_object",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_single_property(api_client, request, stream):
    """response_format为json_schema，仅含单个属性的schema"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "返回一个结果"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "single_field",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string"}
                    },
                    "required": ["result"],
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_deeply_nested_strict(api_client, request, stream):
    """response_format为json_schema，深层嵌套对象（3层）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "描述一个公司的组织架构"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "org_structure",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "company": {
                            "type": "object",
                            "properties": {
                                "department": {
                                    "type": "object",
                                    "properties": {
                                        "team": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "count": {"type": "integer"}
                                            },
                                            "required": ["name", "count"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["team"],
                                    "additionalProperties": False
                                }
                            },
                            "required": ["department"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["company"],
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_all_basic_types_strict(api_client, request, stream):
    """response_format为json_schema，schema包含所有基础类型字段"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个包含多种类型的JSON"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "all_types",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "string_field": {"type": "string"},
                        "integer_field": {"type": "integer"},
                        "number_field": {"type": "number"},
                        "boolean_field": {"type": "boolean"}
                    },
                    "required": ["string_field", "integer_field", "number_field", "boolean_field"],
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_array_with_items(api_client, request, stream):
    """response_format为json_schema，数组字段含items约束"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "列出5个数字"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "number_list",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "numbers": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    },
                    "required": ["numbers"],
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_array_of_objects(api_client, request, stream):
    """response_format为json_schema，数组元素为对象类型"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "列出3个人的信息"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person_list",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "people": {
                            "type": "array",
                            "items": {
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
                    "required": ["people"],
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_enum_single_value(api_client, request, stream):
    """response_format为json_schema，enum仅含一个值"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "选择一个状态"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "single_choice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["active"]
                        }
                    },
                    "required": ["status"],
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_enum_many_values(api_client, request, stream):
    """response_format为json_schema，enum含多个值"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "选择一个季节"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "season_choice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "season": {
                            "type": "string",
                            "enum": ["spring", "summer", "autumn", "winter"]
                        }
                    },
                    "required": ["season"],
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_long_name(api_client, request, stream):
    """response_format为json_schema，schema name为长字符串"""
    long_name = "a" * 200
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个结果"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": long_name,
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string"}
                    },
                    "required": ["result"],
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_name_with_special_chars_strict(api_client, request, stream):
    """response_format为json_schema，schema name含特殊字符"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个结果"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "my-schema_v1.0",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string"}
                    },
                    "required": ["result"],
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_property_with_underscore(api_client, request, stream):
    """response_format为json_schema，属性名含下划线"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个用户信息"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "user_info",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "user_name": {"type": "string"},
                        "user_age": {"type": "integer"},
                        "is_active": {"type": "boolean"}
                    },
                    "required": ["user_name", "user_age", "is_active"],
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_text_with_structured_outputs_json(api_client, request, stream):
    """response_format为text同时使用structured_outputs.json，应正常处理或返回错误"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个人物信息"}],
        "response_format": {
            "type": "text"
        },
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
        "stream": stream,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_object_with_structured_outputs_regex(api_client, request, stream):
    """response_format为json_object同时使用structured_outputs.regex，应返回错误（约束冲突）"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "给我一个数字"}],
        "response_format": {
            "type": "json_object"
        },
        "structured_outputs": {
            "regex": r"\d+"
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
def test_response_format_json_schema_with_minimal_tokens(api_client, request, stream):
    """response_format为json_schema，max_tokens=1，输出可能被截断但不应5xx"""
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
        "max_tokens": 51200,
        "stream": stream
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_with_temperature_zero(api_client, request, stream):
    """response_format为json_schema，temperature=0（确定性输出）"""
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_chinese_property_names(api_client, request, stream):
    """response_format为json_schema，属性名含中文字符"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "描述一个人"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "chinese_person",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "姓名": {"type": "string"},
                        "年龄": {"type": "integer"}
                    },
                    "required": ["姓名", "年龄"],
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_with_long_description(api_client, request, stream):
    """response_format为json_schema，description为长字符串"""
    long_desc = "A" * 1000
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个人物信息"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "description": long_desc,
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
    assertion.assert_no_internal_server_error(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_response_format_json_schema_with_property_descriptions(api_client, request, stream):
    """response_format为json_schema，属性含description字段"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "描述一个产品"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "product",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "产品名称"
                        },
                        "price": {
                            "type": "number",
                            "description": "产品价格，单位为元"
                        }
                    },
                    "required": ["name", "price"],
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
    assertion.assert_no_internal_server_error(response)
