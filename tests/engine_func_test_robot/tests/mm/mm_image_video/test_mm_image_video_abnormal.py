"""多模态对话 - 图片+视频输入 - 异常场景测试"""
import pytest
import allure
from engine_func_test_robot.utility import assertion
from engine_func_test_robot.utility import mm_helper


@allure.feature("多模态对话")
@allure.story("图片+视频输入")
@allure.title("超出最大数量图片输入")
def test_exceed_max_images(api_client, request):
    """测试超出模型支持的最大图片数量"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    if image_num < 1 or video_num < 1:
        pytest.skip("模型不支持图片或视频输入")
    
    # 超出最大数量1个
    images = mm_helper.get_random_images(image_num + 1)
    videos = mm_helper.get_random_videos(1)
    if len(images) < image_num + 1 or not videos:
        pytest.skip(f"没有足够的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, source_type="base64")]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_not_500(response)


@allure.feature("多模态对话")
@allure.story("图片+视频输入")
@allure.title("超出最大数量视频输入")
def test_exceed_max_videos(api_client, request):
    """测试超出模型支持的最大视频数量"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    if image_num < 1 or video_num < 1:
        pytest.skip("模型不支持图片或视频输入")
    
    # 超出最大数量1个
    images = mm_helper.get_random_images(1)
    videos = mm_helper.get_random_videos(video_num + 1)
    if not images or len(videos) < video_num + 1:
        pytest.skip(f"没有足够的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, source_type="base64")]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_not_500(response)


@allure.feature("多模态对话")
@allure.story("图片+视频输入")
@allure.title("无效图片URL")
def test_invalid_image_url(api_client, request):
    """测试无效的图片URL"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    if image_num < 1 or video_num < 1:
        pytest.skip("模型不支持图片或视频输入")
    
    video_urls = mm_helper.get_random_video_urls(1)
    if not video_urls:
        pytest.skip("没有可用的测试视频URL")
    
    prompt = "请描述这张图片和这个视频的内容"
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": "https://invalid-url-that-does-not-exist.com/image.png"}},
        {"type": "video_url", "video_url": {"url": video_urls[0]}}
    ]
    messages = [{"role": "user", "content": content}]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)


@allure.feature("多模态对话")
@allure.story("图片+视频输入")
@allure.title("无效视频URL")
def test_invalid_video_url(api_client, request):
    """测试无效的视频URL"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    if image_num < 1 or video_num < 1:
        pytest.skip("模型不支持图片或视频输入")
    
    image_urls = mm_helper.get_random_image_urls(1)
    if not image_urls:
        pytest.skip("没有可用的测试图片URL")
    
    prompt = "请描述这张图片和这个视频的内容"
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_urls[0]}},
        {"type": "video_url", "video_url": {"url": "https://invalid-url-that-does-not-exist.com/video.mp4"}}
    ]
    messages = [{"role": "user", "content": content}]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)


@allure.feature("多模态对话")
@allure.story("图片+视频输入")
@allure.title("无效图片Base64")
def test_invalid_image_base64(api_client, request):
    """测试无效的图片Base64数据"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    if image_num < 1 or video_num < 1:
        pytest.skip("模型不支持图片或视频输入")
    
    videos = mm_helper.get_random_videos(1)
    if not videos:
        pytest.skip("没有可用的测试视频")
    
    # 将视频转换为base64格式
    video_content = mm_helper.build_video_content(videos[0], source_type="base64")
    video_url = video_content["video_url"]["url"]
    
    prompt = "请描述这张图片和这个视频的内容"
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": "image/png;base64,invalid_base64_data"}},
        {"type": "video_url", "video_url": {"url": video_url}}
    ]
    messages = [{"role": "user", "content": content}]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)


@allure.feature("多模态对话")
@allure.story("图片+视频输入")
@allure.title("无效视频Base64")
def test_invalid_video_base64(api_client, request):
    """测试无效的视频Base64数据"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    if image_num < 1 or video_num < 1:
        pytest.skip("模型不支持图片或视频输入")
    
    images = mm_helper.get_random_images(1)
    if not images:
        pytest.skip("没有可用的测试图片")
    
    # 将图片转换为base64格式
    image_content = mm_helper.build_image_content(images[0], source_type="base64")
    image_url = image_content["image_url"]["url"]
    
    prompt = "请描述这张图片和这个视频的内容"
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "video_url", "video_url": {"url": "video/mp4;base64,invalid_base64_data"}}
    ]
    messages = [{"role": "user", "content": content}]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)


@allure.feature("多模态对话")
@allure.story("图片+视频输入")
@allure.title("空图片URL")
def test_empty_image_url(api_client, request):
    """测试空的图片URL"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    if image_num < 1 or video_num < 1:
        pytest.skip("模型不支持图片或视频输入")
    
    video_urls = mm_helper.get_random_video_urls(1)
    if not video_urls:
        pytest.skip("没有可用的测试视频URL")
    
    prompt = "请描述这张图片和这个视频的内容"
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": ""}},
        {"type": "video_url", "video_url": {"url": video_urls[0]}}
    ]
    messages = [{"role": "user", "content": content}]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)


@allure.feature("多模态对话")
@allure.story("图片+视频输入")
@allure.title("空视频URL")
def test_empty_video_url(api_client, request):
    """测试空的视频URL"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    if image_num < 1 or video_num < 1:
        pytest.skip("模型不支持图片或视频输入")
    
    image_urls = mm_helper.get_random_image_urls(1)
    if not image_urls:
        pytest.skip("没有可用的测试图片URL")
    
    prompt = "请描述这张图片和这个视频的内容"
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_urls[0]}},
        {"type": "video_url", "video_url": {"url": ""}}
    ]
    messages = [{"role": "user", "content": content}]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)