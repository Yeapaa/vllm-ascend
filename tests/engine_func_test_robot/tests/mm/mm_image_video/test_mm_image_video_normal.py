"""多模态对话 - 图片+视频输入 - 正常场景测试"""
import pytest
import allure
from engine_func_test_robot.utility import assertion
from engine_func_test_robot.utility import mm_helper


@allure.feature("多模态对话")
@allure.story("图片+视频输入")
@allure.title("单图片单视频输入-Base64")
def test_single_image_single_video_base64(api_client, request):
    """测试单图片单视频输入（Base64方式）"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    if image_num < 1 or video_num < 1:
        pytest.skip("模型不支持图片或视频输入")
    
    images = mm_helper.get_random_images(1)
    videos = mm_helper.get_random_videos(1)
    if not images or not videos:
        pytest.skip("没有可用的测试图片或视频")
    
    prompt = "请描述这张图片和这个视频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, source_type="base64")]
    
    mm_helper.attach_source_files(images=images, videos=videos, source_type="base64")
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_status_code_200(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@allure.feature("多模态对话")
@allure.story("图片+视频输入")
@allure.title("单图片单视频输入-URL")
def test_single_image_single_video_url(api_client, request):
    """测试单图片单视频输入（URL方式）"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    if image_num < 1 or video_num < 1:
        pytest.skip("模型不支持图片或视频输入")
    
    images = mm_helper.get_random_image_urls(1)
    videos = mm_helper.get_random_video_urls(1)
    if not images or not videos:
        pytest.skip("没有可用的测试图片或视频URL")
    
    prompt = "请描述这张图片和这个视频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, source_type="url")]
    
    mm_helper.attach_source_files(images=images, videos=videos, source_type="url")
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_status_code_200(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@allure.feature("多模态对话")
@allure.story("图片+视频输入")
@allure.title("最大数量图片和视频输入-Base64")
def test_max_images_and_videos_base64(api_client, request):
    """测试最大数量的图片和视频输入（Base64方式）"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    if image_num < 1 or video_num < 1:
        pytest.skip("模型不支持图片或视频输入")
    
    images = mm_helper.get_random_images(image_num)
    videos = mm_helper.get_random_videos(video_num)
    if len(images) < image_num or len(videos) < video_num:
        pytest.skip(f"没有足够的测试资源，需要{image_num}张图片和{video_num}个视频")
    
    prompt = f"请描述这{image_num}张图片和{video_num}个视频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, source_type="base64")]
    
    mm_helper.attach_source_files(images=images, videos=videos, source_type="base64")
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_status_code_200(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@allure.feature("多模态对话")
@allure.story("图片+视频输入")
@allure.title("最大数量图片和视频输入-URL")
def test_max_images_and_videos_url(api_client, request):
    """测试最大数量的图片和视频输入（URL方式）"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    if image_num < 1 or video_num < 1:
        pytest.skip("模型不支持图片或视频输入")
    
    images = mm_helper.get_random_image_urls(image_num)
    videos = mm_helper.get_random_video_urls(video_num)
    if len(images) < image_num or len(videos) < video_num:
        pytest.skip(f"没有足够的测试资源URL，需要{image_num}张图片和{video_num}个视频")
    
    prompt = f"请描述这{image_num}张图片和{video_num}个视频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, source_type="url")]
    
    mm_helper.attach_source_files(images=images, videos=videos, source_type="url")
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_status_code_200(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@allure.feature("多模态对话")
@allure.story("图片+视频输入")
@allure.title("图片视频混合输入-Base64和URL")
def test_mixed_source_image_video(api_client, request):
    """测试图片视频混合输入（Base64和URL混合）"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    if image_num < 2 or video_num < 2:
        pytest.skip("模型不支持足够的图片或视频输入")
    
    # 获取Base64和URL各一个
    images_base64 = mm_helper.get_random_images(1)
    images_url = mm_helper.get_random_images(1, source_type="url")
    videos_base64 = mm_helper.get_random_videos(1)
    videos_url = mm_helper.get_random_videos(1, source_type="url")
    
    if not all([images_base64, images_url, videos_base64, videos_url]):
        pytest.skip("没有足够的测试资源")
    
    prompt = "请描述这些图片和视频的内容"
    
    # 构建混合内容
    content = [{"type": "text", "text": prompt}]
    
    # 添加Base64图片
    content.append({
        "type": "image_url",
        "image_url": {"url": images_base64[0]}
    })
    
    # 添加URL图片
    content.append({
        "type": "image_url",
        "image_url": {"url": images_url[0]}
    })
    
    # 添加Base64视频
    content.append({
        "type": "video_url",
        "video_url": {"url": videos_base64[0]}
    })
    
    # 添加URL视频
    content.append({
        "type": "video_url",
        "video_url": {"url": videos_url[0]}
    })
    
    messages = [{"role": "user", "content": content}]
    
    mm_helper.attach_source_files(images=images_base64 + images_url, videos=videos_base64 + videos_url, source_type="mixed")
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_status_code_200(response)
    finish_reason = response.json()["choices"][0]["finish_reason"]
    assertion.assert_finish_reason_valid(finish_reason)


@allure.feature("多模态对话")
@allure.story("图片+视频输入")
@allure.title("图片视频输入-流式响应")
def test_image_video_stream(api_client, request):
    """测试图片视频输入的流式响应"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    if image_num < 1 or video_num < 1:
        pytest.skip("模型不支持图片或视频输入")
    
    images = mm_helper.get_random_images(1)
    videos = mm_helper.get_random_videos(1)
    if not images or not videos:
        pytest.skip("没有可用的测试图片或视频")
    
    prompt = "请描述这张图片和这个视频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, source_type="base64")]
    
    mm_helper.attach_source_files(images=images, videos=videos, source_type="base64")
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500,
        "stream": True
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_status_code_200(response)
    assertion.assert_stream_has_done(response.text)
    finish_reason = assertion.assert_stream_single_finish_reason(response.text)
    assertion.assert_finish_reason_valid(finish_reason)