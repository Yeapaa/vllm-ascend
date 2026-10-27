import re
import json
import pytest_check as check

# think标签定义
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def assert_status_code_200(response, msg=""):
    """校验HTTP状态码为200"""
    check.equal(response.status_code, 200, f"{msg}响应状态码不为200")


def assert_status_code_400(response, msg=""):
    """校验HTTP状态码为400"""
    check.equal(response.status_code, 400, f"{msg}响应状态码不为400")


def assert_finish_reason_stop(finish_reason, msg=""):
    """校验finish_reason为stop"""
    check.equal(finish_reason, "stop", f"{msg}finish_reason不为stop")


def assert_finish_reason_valid(finish_reason, msg=""):
    """校验finish_reason为stop或length"""
    check.is_in(finish_reason, ["stop", "length"], f"{msg}finish_reason不为stop或length")


def assert_stream_has_done(response_text, msg=""):
    """校验流式响应包含[DONE]"""
    check.is_true(
        re.search(r'^data:\s*\[DONE\](?:\n|$)', response_text, re.M),
        f"{msg}流式响应没有[DONE]"
    )


def assert_stream_single_finish_reason(response_text, msg=""):
    """校验流式响应只有一个finish_reason，返回该值"""
    finish_reasons = re.findall(r'finish_reason":\s*"([^"]+)"', response_text, re.M)
    check.equal(len(finish_reasons), 1, f"{msg}流式响应finish_reason存在多个值")
    return finish_reasons[0] if finish_reasons else None


def _parse_stream_chunks(response_text):
    """解析流式响应中的data chunk，返回有效chunk的json列表"""
    chunks = []
    for chunk_str in re.findall(r'^data:\s*(\{.*\})\s*$', response_text, re.M):
        try:
            chunks.append(json.loads(chunk_str))
        except json.JSONDecodeError:
            continue
    return chunks


def assert_stream_finish_reason_valid(response_text, msg=""):
    """校验流式响应：最后一个data chunk的finish_reason为stop或length，其余为空字符串或None"""
    chunks = _parse_stream_chunks(response_text)
    check.is_true(len(chunks) > 0, f"{msg}流式响应无有效data chunk")

    for idx, chunk_json in enumerate(chunks):
        choices = chunk_json.get("choices", [])
        if not choices:
            continue
        finish_reason = choices[0].get("finish_reason")
        is_last = (idx == len(chunks) - 1)

        if is_last:
            check.is_in(finish_reason, ["stop", "length"],
                        f"{msg}最后一个chunk的finish_reason应为stop或length，实际为: {finish_reason}")
        else:
            check.is_true(finish_reason in ("", None),
                          f"{msg}第{idx + 1}个chunk的finish_reason应为空，实际为: {finish_reason}")


def assert_stream_thinking_content_order(response_text, msg=""):
    """校验流式思考模式响应：
    1. 除最后一个reasoning chunk外，其他chunk的content和reasoning_content/reasoning不会同时有值
    2. 除最后一个reasoning chunk外，其他reasoning chunk必须出现在content chunk之前
    3. 若最后一个chunk的finish_reason为stop，则一定存在content有值的chunk
    """
    chunks = _parse_stream_chunks(response_text)
    check.is_true(len(chunks) > 0, f"{msg}流式响应无有效data chunk")

    has_content_chunk = False
    found_content_after_reasoning = False
    last_reasoning_chunk_idx = -1

    for idx, chunk_json in enumerate(chunks):
        choices = chunk_json.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content", "") or ""
        reasoning = delta.get("reasoning_content", "") or delta.get("reasoning", "") or ""

        # 记录最后一个有reasoning值的chunk索引
        if len(reasoning) > 0:
            last_reasoning_chunk_idx = idx

        # 校验1：除最后一个reasoning chunk外，其他chunk的content和reasoning不会同时有值
        if idx != last_reasoning_chunk_idx:
            check.is_true(
                not (len(content) > 0 and len(reasoning) > 0),
                f"{msg}第{idx + 1}个chunk的content和reasoning_content/reasoning同时有值"
            )

        # 校验2：除最后一个reasoning chunk外，reasoning chunk必须在content chunk之前
        if idx != last_reasoning_chunk_idx:
            if len(content) > 0:
                has_content_chunk = True
            if len(reasoning) > 0 and has_content_chunk:
                found_content_after_reasoning = True

    check.is_false(found_content_after_reasoning,
                   f"{msg}reasoning_content/reasoning有值的chunk出现在content有值的chunk之后")

    # 校验3：finish_reason为stop时，一定存在content有值的chunk
    last_choices = chunks[-1].get("choices", [])
    if last_choices:
        last_finish_reason = last_choices[0].get("finish_reason")
        if last_finish_reason == "stop":
            check.is_true(has_content_chunk,
                          f"{msg}finish_reason为stop时，不存在content有值的chunk")


def assert_stream_no_reasoning_content(response_text, msg=""):
    """校验流式非思考模式响应：
    1. 每个chunk的reasoning_content/reasoning字段无值或不存在
    2. 一定存在content有值的chunk
    """
    chunks = _parse_stream_chunks(response_text)
    check.is_true(len(chunks) > 0, f"{msg}流式响应无有效data chunk")

    has_content_chunk = False

    for idx, chunk_json in enumerate(chunks):
        choices = chunk_json.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content", "") or ""
        reasoning = delta.get("reasoning_content", "") or delta.get("reasoning", "") or ""

        # 校验1：reasoning_content/reasoning无值
        check.is_true(len(reasoning) == 0,
                      f"{msg}第{idx + 1}个chunk的reasoning_content/reasoning应为空，实际有值")

        if len(content) > 0:
            has_content_chunk = True

    # 校验2：存在content有值的chunk
    check.is_true(has_content_chunk, f"{msg}不存在content有值的chunk")


def assert_think_tag_present(response_text, msg=""):
    """校验存在完整的think标签对"""
    think_open_count = response_text.count(THINK_OPEN)
    think_close_count = response_text.count(THINK_CLOSE)
    check.equal(think_open_count, think_close_count, f"{msg}think标签不完整, OPEN: {think_open_count}, CLOSE: {think_close_count}")
    check.equal(think_open_count, 1, f"{msg}没有think标签")


def assert_no_think_tag(response_text, msg=""):
    """校验不存在think标签"""
    check.equal(response_text.count(THINK_OPEN), 0, f"{msg}存在think标签")


def assert_json_response_content(response_text, msg=""):
    """校验响应内容为有效JSON（过滤think标签后）"""
    pattern = rf"\s*{re.escape(THINK_OPEN)}[\s\S]*?{re.escape(THINK_CLOSE)}"
    json_str = re.sub(pattern, "", response_text)
    match = re.search(r"(\{.*\})\s*(?:$|`|```)$", json_str, re.S)
    check.is_true(match, f"{msg}内容非JSON格式")
    if match:
        json.loads(match.group(1))


def validate_json_response(response, stream):
    """JSON响应正确性校验：finish_reason为stop时校验响应内容为有效JSON

    非流式：从response.json()取各choice的message.content，逐个做JSON校验
    流式：解析SSE chunks拼接各choice的delta.content，逐个做JSON校验
    """
    if not stream:
        resp_json = response.json()
        for choice in resp_json.get("choices", []):
            finish_reason = choice.get("finish_reason")
            assert_finish_reason_stop(finish_reason, f"响应choices[{choice.get('index', 0)}]finish_reason验证错误")
            if finish_reason == "stop":
                content = choice.get("message", {}).get("content", "") or ""
                assert_json_response_content(
                    content, f"响应choices[{choice.get('index', 0)}]内容json验证错误"
                )
    else:
        response_text = response.text
        chunks = _parse_stream_chunks(response_text)
        if chunks:
            last_choices = chunks[-1].get("choices", [])
            if last_choices:
                finish_reason = last_choices[0].get("finish_reason")
                assert_finish_reason_stop(finish_reason, f"响应finish_reason验证错误")
                if finish_reason == "stop":
                    choices_content = {}
                    for chunk in chunks:
                        for choice in chunk.get("choices", []):
                            idx = choice.get("index", 0)
                            delta = choice.get("delta", {})
                            content = delta.get("content", "") or ""
                            choices_content.setdefault(idx, "")
                            choices_content[idx] += content
                    for idx in sorted(choices_content.keys()):
                        assert_json_response_content(
                            choices_content[idx], f"响应choices[{idx}]内容json验证错误"
                        )


def has_error_code(response):
    """判断响应中是否存在error code"""
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        response_json = response.json()
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        return error_code is not None
    elif "text/event-stream" in content_type or "text/plain" in content_type:
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        return match is not None
    return False


def assert_no_error_code(response, msg=""):
    """校验响应中不存在error code字段"""
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        response_json = response.json()
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        check.is_none(error_code, f"{msg}响应体中存在error code，不允许存在code字段")
    elif "text/event-stream" in content_type or "text/plain" in content_type:
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        check.is_none(match, f"{msg}流式响应中存在code字段，不允许存在code字段")


def assert_error_code_in(response, expected_codes, msg=""):
    """校验错误码在指定集合内"""
    if "application/json" in response.headers.get("Content-Type", ""):
        error_code = response.json().get("error", {}).get("code") or response.json().get("code")
        check.is_in(error_code, expected_codes, f"{msg}错误码不在期望集合{expected_codes}中，实际: {error_code}")
    elif "text/event-stream" in response.headers.get("Content-Type", ""):
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match:
            check.is_in(int(match.group(1)), expected_codes, f"{msg}流式响应错误码不在期望集合{expected_codes}中")


def assert_error_code_400(response, msg=""):
    """校验错误码为400"""
    if "application/json" in response.headers.get("Content-Type", ""):
        error_code = response.json().get("error", {}).get("code") or response.json().get("code")
        check.equal(error_code, 400, f"{msg}错误码不为400")
    elif "text/event-stream" in response.headers.get("Content-Type", ""):
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match:
            check.equal(int(match.group(1)), 400, f"{msg}流式响应错误码不为400")


def assert_error_code_422(response, msg=""):
    """校验错误码为422 Unprocessable Entity（数据验证失败）"""
    if "application/json" in response.headers.get("Content-Type", ""):
        error_code = response.json().get("error", {}).get("code") or response.json().get("code")
        check.equal(error_code, 422, f"{msg}错误码不为422（Unprocessable Entity-数据验证失败）")
    elif "text/event-stream" in response.headers.get("Content-Type", ""):
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match:
            check.equal(int(match.group(1)), 422, f"{msg}流式响应错误码不为422")


def assert_error_code_not_5xx(response, msg=""):
    """校验如果响应体存在error code，则error code不以5开头（非5xx）"""
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        error_code = response.json().get("error", {}).get("code") or response.json().get("code")
        if error_code is not None:
            check.is_false(str(error_code).startswith("5"), f"{msg}error code不应为5xx，实际: {error_code}")
    elif "text/event-stream" in content_type:
        match = re.search(r'\"code\"\s?:\s?(\d+)', response.text, re.M)
        if match:
            error_code = int(match.group(1))
            check.is_false(str(error_code).startswith("5"), f"{msg}error code不应为5xx，实际: {error_code}")


def assert_no_internal_server_error(response, msg=""):
    """校验响应中不包含Internal Server Error（忽略大小写和所有空白字符）"""
    response_text = response.text
    normalized = re.sub(r'\s+', '', response_text).lower()
    check.is_false(
        'internalservererror' in normalized,
        f"{msg}响应中包含Internal Server Error"
    )


def assert_image_edit_response_fields(response, msg=""):
    """校验图像编辑接口响应字段完整性

    Args:
        response: HTTP响应对象
        msg: 错误消息前缀
    """
    resp_json = response.json()

    # 校验顶层字段
    check.is_true("created" in resp_json, f"{msg}响应应包含created字段")
    check.is_true("data" in resp_json, f"{msg}响应应包含data字段")
    check.is_true("output_format" in resp_json, f"{msg}响应应包含output_format字段")
    check.is_true("size" in resp_json, f"{msg}响应应包含size字段")

    # 校验data数组
    data = resp_json.get("data", [])
    check.is_true(len(data) > 0, f"{msg}data应包含至少一个结果")

    # 校验data数组元素字段
    for idx, item in enumerate(data):
        has_b64 = "b64_json" in item and item["b64_json"]
        has_url = "url" in item and item["url"]
        check.is_true(has_b64 or has_url, f"{msg}data[{idx}]应包含b64_json或url字段")
        check.is_true("revised_prompt" in item, f"{msg}data[{idx}]应包含revised_prompt字段")

    return resp_json


def assert_top_logprobs_count(response, top_logprobs_value, msg=""):
    """校验logprobs的top_logprobs数量"""
    content_type = response.headers.get("Content-Type", "")

    if "application/json" in content_type:
        logprobs_content_list = response.json()["choices"][0]["logprobs"]["content"]
        for item_dict in logprobs_content_list:
            check.equal(
                len(item_dict.get("top_logprobs")),
                top_logprobs_value,
                f"{msg}logprobs top_logprobs长度不为{top_logprobs_value}"
            )
    elif "text/event-stream" in content_type:
        chunk_list = re.findall(r'^data:\s*(.*)(?:\n|$)', response.text, re.M)[1:-1]
        for chunk_item in chunk_list:
            chunk_json = json.loads(chunk_item)
            content = chunk_json["choices"][0]["delta"].get("content", "")    # todo⚠️：content为空时，top_logprobs为None,没有值。需要确认 or <|im_end|>...。即如下 if xxx else yyy 的 else 分支
            if content:
                logprobs_content_list = chunk_json["choices"][0]["logprobs"]["content"]
                for item_dict in logprobs_content_list:
                    check.equal(
                        len(item_dict.get("top_logprobs")),
                        top_logprobs_value,
                        f"{msg}流式logprobs top_logprobs长度不为{top_logprobs_value}"
                    )
