import pytest
import json
from ...utility import request_helper as helper
from ...utility import assertion


SIMPLE_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"],
    "additionalProperties": False
}


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_json_dict(api_client, request, stream):
    """structured_outputs.json为dict类型，正常生成结构化输出，vllm/sglang并不支持该field结构化输出"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个人物信息"}],
        "structured_outputs": {
            "json": SIMPLE_OBJECT_SCHEMA
        },
        "stream": stream,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_json_string(api_client, request, stream):
    """structured_outputs.json为字符串类型，正常生成结构化输出，vllm/sglang并不支持该field结构化输出"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个人物信息"}],
        "structured_outputs": {
            "json": json.dumps(SIMPLE_OBJECT_SCHEMA)
        },
        "stream": stream,
        "max_tokens": 51200
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_regex(api_client, request, stream):
    """structured_outputs.regex，正常生成符合正则的输出"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "给我一个手机号"}],
        "structured_outputs": {
            "regex": r"1[3-9]\d{9}"
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_choice(api_client, request, stream):
    """structured_outputs.choice，正常生成选项之一的输出"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "今天心情怎么样？"}],
        "structured_outputs": {
            "choice": ["开心", "难过", "平静"]
        },
        "stream": stream,
        "max_tokens": 512
    }

    helper.attach_request_body(request_body)
    response = helper.send_request(api_client, "/v1/chat/completions", request_body)
    helper.attach_response_body(response)

    assertion.assert_error_code_not_5xx(response)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_grammar(api_client, request, stream):
    """structured_outputs.grammar，使用EBNF语法约束输出"""
    grammar = r'''
root ::= answer
answer ::= "yes" | "no"
'''
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "你喜欢编程吗？"}],
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


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_structured_outputs_json_object(api_client, request, stream):
    """structured_outputs.json_object为true，正常生成JSON对象输出"""
    request_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "生成一个包含name和age字段的JSON对象"}],
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
    assertion.validate_json_response(response, stream)
