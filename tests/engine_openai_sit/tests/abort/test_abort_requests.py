import http.client
import json
import random
import socket
import time

import allure
import pytest

from ...utility import assertion
from ...utility import request_helper as helper
from ...utility.http_client import HTTPClient


def _get_node_client(request, option_name, node_name):
    """根据命令行参数创建指定节点的HTTP客户端。"""
    node_address = request.config.getoption(option_name).strip()
    if not node_address:
        pytest.skip(f"未指定{node_name}节点地址：{option_name}")
    if not node_address.startswith(("http://", "https://")):
        node_address = f"http://{node_address}"
    return HTTPClient(base_url=node_address)


def _get_remote_request_id(response_json):
    """兼容从remote_request_id或transfer_id获取远端请求ID。"""
    kv_transfer_params = response_json.get("kv_transfer_params", {})
    remote_request_id = kv_transfer_params.get("remote_request_id")
    if not isinstance(remote_request_id, str) or not remote_request_id:
        transfer_id = kv_transfer_params.get("transfer_id")
        if isinstance(transfer_id, str):
            remote_request_id = transfer_id.removeprefix("xfer-")

    assert isinstance(remote_request_id, str) and remote_request_id, \
        "响应体kv_transfer_params.remote_request_id或transfer_id不存在或为空"
    return remote_request_id


def _send_inference_and_abort(node_client, endpoint, kv_transfer_params):
    """发送非流式请求，使用响应中的远端请求ID发起abort请求。"""
    request_body = {
        "model": "auto",
        "stream": False,
        "kv_transfer_params": kv_transfer_params,
        "max_tokens": 20480
    }
    if endpoint == "/v1/chat/completions":
        request_body["messages"] = [{
            "role": "user",
            "content": "请详细介绍人工智能的发展历史。"
        }]
    else:
        request_body["prompt"] = "请详细介绍人工智能的发展历史。"

    helper.attach_request_body(request_body, name=f"{endpoint}请求体")
    response = helper.send_request(node_client, endpoint, request_body)
    helper.attach_response_body(response, name=f"{endpoint}响应体")

    remote_request_id = _get_remote_request_id(response.json())

    with allure.step("等待3秒后发送Abort请求"):
        time.sleep(3)

    abort_request_body = {
        "request_ids": [remote_request_id]
    }
    helper.attach_request_body(abort_request_body, name="Abort请求体")
    abort_response = helper.send_request(node_client, "/abort_requests", abort_request_body)
    helper.attach_response_body(abort_response, name="Abort响应体")

    assertion.assert_status_code_200(abort_response, msg="Abort请求：")
    assertion.assert_no_error_code(abort_response, msg="Abort请求：")


def _send_and_disconnect(request, endpoint, stream, prompt_repeat):
    """发送推理请求，随机等待1至3秒后主动关闭底层连接。"""
    host = request.config.getoption("--modelServiceIP").strip()
    port = int(request.config.getoption("--modelServicePort"))
    prompt = "请详细介绍人工智能的发展历史。" * prompt_repeat
    request_body = {
        "model": "auto",
        "stream": stream,
        "max_tokens": 20480
    }
    if endpoint == "/v1/chat/completions":
        request_body["messages"] = [{
            "role": "user",
            "content": prompt
        }]
    else:
        request_body["prompt"] = prompt

    helper.attach_request_body(request_body, name=f"{endpoint}异常中断请求体")
    request_data = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    connection = http.client.HTTPConnection(host, port, timeout=10)

    try:
        connection.request(
            "POST",
            endpoint,
            body=request_data,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(request_data))
            }
        )

        wait_seconds = random.uniform(1, 3)
        with allure.step(f"请求发送后等待{wait_seconds:.3f}秒"):
            time.sleep(wait_seconds)
    finally:
        with allure.step("主动关闭推理请求连接，模拟客户端异常中断"):
            if connection.sock is not None:
                try:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()


@pytest.mark.parametrize(
    "endpoint",
    ["/v1/chat/completions", "/v1/completions"],
    ids=["chat_completions", "completions"]
)
def test_abort_prefill_request(request, endpoint):
    """向prefill节点发起远程decode请求，并按远端请求ID执行abort。"""
    prefill_client = _get_node_client(
        request,
        "--prefillNodeAddress",
        "prefill"
    )
    _send_inference_and_abort(
        prefill_client,
        endpoint,
        kv_transfer_params={
            "do_remote_decode": True,
            "do_remote_prefill": False
        }
    )


@pytest.mark.parametrize(
    "endpoint",
    ["/v1/chat/completions", "/v1/completions"],
    ids=["chat_completions", "completions"]
)
def test_abort_decode_request(request, endpoint):
    """向decode节点发起远程prefill请求，并按远端请求ID执行abort。"""
    decode_client = _get_node_client(
        request,
        "--decodeNodeAddress",
        "decode"
    )
    _send_inference_and_abort(
        decode_client,
        endpoint,
        kv_transfer_params={
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "last_token_id": 27620
        }
    )


@pytest.mark.parametrize(
    "endpoint",
    ["/v1/chat/completions", "/v1/completions"],
    ids=["chat_completions", "completions"]
)
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("prompt_repeat", [1, 7000], ids=["short_prompt", "long_prompt_7000x"])
def test_inference_client_disconnect(request, endpoint, stream, prompt_repeat):
    """向模型服务发送推理请求，随机等待1至3秒后模拟客户端异常中断。"""
    _send_and_disconnect(request, endpoint, stream, prompt_repeat)
