"""多模态对话 - 图片输入 - 正常场景测试"""
import pytest
import allure
from engine_func_test_robot.utility import assertion
from engine_func_test_robot.utility import mm_helper


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("单张图片URL输入")
def test_single_image_url(api_client, request):
    """测试单张图片URL输入"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 1:
        pytest.skip("模型不支持图片输入")
    
    image_urls = mm_helper.get_random_image_urls(1)
    if not image_urls:
        pytest.skip("没有可用的图片URL")
    
    prompt = "请描述这张图片的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=image_urls, source_type="url")]
    
    mm_helper.attach_source_files(images=image_urls, source_type="url")
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_status_code_200(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("单张图片Base64输入")
def test_single_image_base64(api_client, request):
    """测试单张图片Base64输入"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 1:
        pytest.skip("模型不支持图片输入")
    
    images = mm_helper.get_random_images(1)
    if not images:
        pytest.skip("没有可用的测试图片")
    
    prompt = "请描述这张图片的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, source_type="base64")]
    
    mm_helper.attach_source_files(images=images, source_type="base64")
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_status_code_200(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("多张图片URL输入")
def test_multiple_image_urls(api_client, request):
    """测试多张图片URL输入"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 2:
        pytest.skip(f"模型最多支持{image_num}张图片输入")
    
    image_urls = mm_helper.get_random_image_urls(image_num)
    if not image_urls:
        pytest.skip("没有可用的图片URL")
    
    prompt = "请描述这些图片的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=image_urls, source_type="url")]
    
    mm_helper.attach_source_files(images=image_urls, source_type="url")
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_status_code_200(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("多张图片Base64输入")
def test_multiple_image_base64(api_client, request):
    """测试多张图片Base64输入"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 2:
        pytest.skip(f"模型最多支持{image_num}张图片输入")
    
    images = mm_helper.get_random_images(image_num)
    if not images:
        pytest.skip("没有可用的测试图片")
    
    prompt = "请描述这些图片的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, source_type="base64")]
    
    mm_helper.attach_source_files(images=images, source_type="base64")
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_status_code_200(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("图片URL输入流式响应")
def test_image_url_stream(api_client, request):
    """测试图片URL输入流式响应"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 1:
        pytest.skip("模型不支持图片输入")
    
    image_urls = mm_helper.get_random_image_urls(1)
    if not image_urls:
        pytest.skip("没有可用的图片URL")
    
    prompt = "请描述这张图片的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=image_urls, source_type="url")]
    
    mm_helper.attach_source_files(images=image_urls, source_type="url")
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100,
        "stream": True
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_status_code_200(response)
    assertion.assert_stream_has_done(response.text)
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("图片Base64输入流式响应")
def test_image_base64_stream(api_client, request):
    """测试图片Base64输入流式响应"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 1:
        pytest.skip("模型不支持图片输入")
    
    images = mm_helper.get_random_images(1)
    if not images:
        pytest.skip("没有可用的测试图片")
    
    prompt = "请描述这张图片的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, source_type="base64")]
    
    mm_helper.attach_source_files(images=images, source_type="base64")
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100,
        "stream": True
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_status_code_200(response)
    assertion.assert_stream_has_done(response.text)
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("图片URL输入多轮对话")
def test_image_url_multi_turn(api_client, request):
    """测试图片URL输入多轮对话"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 1:
        pytest.skip("模型不支持图片输入")
    
    image_urls = mm_helper.get_random_image_urls(1)
    if not image_urls:
        pytest.skip("没有可用的图片URL")
    
    # 第一轮
    messages = [mm_helper.build_multimodal_message("请描述这张图片的内容", images=image_urls, source_type="url")]
    
    mm_helper.attach_source_files(images=image_urls, source_type="url")
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100
    }
    
    mm_helper.attach_multimodal_request(request_body, "第一轮请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "第一轮响应体")
    
    assertion.assert_status_code_200(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)
    
    # 获取第一轮响应
    first_response = response.json()
    first_answer = first_response["choices"][0]["message"]["content"]
    
    # 第二轮
    messages.append({"role": "assistant", "content": first_answer})
    messages.append({"role": "user", "content": "请详细说明"})
    
    request_body["messages"] = messages
    
    mm_helper.attach_multimodal_request(request_body, "第二轮请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "第二轮响应体")
    
    assertion.assert_status_code_200(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)