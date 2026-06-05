"""多模态对话 - 图片输入 - 异常场景测试"""
import pytest
import allure
from engine_func_test_robot.utility import assertion
from engine_func_test_robot.utility import mm_helper


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("无效图片URL")
def test_invalid_image_url(api_client, request):
    """测试无效图片URL"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 1:
        pytest.skip("模型不支持图片输入")
    
    invalid_urls = ["https://invalid-url-that-does-not-exist.com/image.jpg", "not_a_url"]
    
    prompt = "请描述这张图片的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=invalid_urls, source_type="url")]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("无效图片Base64")
def test_invalid_image_base64(api_client, request):
    """测试无效图片Base64"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 1:
        pytest.skip("模型不支持图片输入")
    
    invalid_base64 = ["invalid_base64_string", ""]
    
    prompt = "请描述这张图片的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=invalid_base64, source_type="base64")]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("超出最大图片数量")
def test_exceed_max_image_count(api_client, request):
    """测试超出最大图片数量"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 1:
        pytest.skip("模型不支持图片输入")
    
    # 超出最大数量1张
    exceed_count = image_num + 1
    images = mm_helper.get_random_images(exceed_count)
    if not images:
        pytest.skip("没有可用的测试图片")
    
    prompt = "请描述这些图片的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, source_type="base64")]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_not_500(response)


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("空图片数组")
def test_empty_image_array(api_client, request):
    """测试空图片数组"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 1:
        pytest.skip("模型不支持图片输入")
    
    prompt = "请描述这张图片的内容"
    # 构建一个包含空图片URL的消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": ""}}
            ]
        }
    ]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("不支持的图片格式URL")
def test_unsupported_image_format_url(api_client, request):
    """测试不支持的图片格式URL"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 1:
        pytest.skip("模型不支持图片输入")
    
    # 使用不支持的格式URL
    unsupported_urls = [
        "https://example.com/document.pdf",
        "https://example.com/video.mp4"
    ]
    
    prompt = "请描述这张图片的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=unsupported_urls, source_type="url")]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("图片URL格式错误")
def test_malformed_image_url(api_client, request):
    """测试图片URL格式错误"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 1:
        pytest.skip("模型不支持图片输入")
    
    malformed_urls = [
        "htp://invalid-protocol.com/image.jpg",
        "://missing-protocol.com/image.jpg",
        "http:///missing-host/image.jpg"
    ]
    
    prompt = "请描述这张图片的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=malformed_urls, source_type="url")]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("图片Base64格式错误")
def test_malformed_image_base64(api_client, request):
    """测试图片Base64格式错误"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 1:
        pytest.skip("模型不支持图片输入")
    
    # 错误的Base64格式
    malformed_base64 = [
        "image/png;base64,invalid!!!",
        "data:image/jpeg;base64,",
        "not_valid_base64_at_all!!!###"
    ]
    
    prompt = "请描述这张图片的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=malformed_base64, source_type="base64")]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)


@allure.feature("多模态对话")
@allure.story("图片输入")
@allure.title("图片内容类型错误")
def test_invalid_image_content_type(api_client, request):
    """测试图片内容类型错误"""
    image_num = request.config.getoption("--imageNum")
    if image_num < 1:
        pytest.skip("模型不支持图片输入")
    
    # 使用错误的内容类型前缀
    invalid_content_type = [
        "video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28y",
        "data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA"
    ]
    
    prompt = "请描述这张图片的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=invalid_content_type, source_type="base64")]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 100
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)