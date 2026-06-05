"""多模态对话 - 视频输入 - 异常场景测试"""
import pytest
import allure
from engine_func_test_robot.utility import assertion
from engine_func_test_robot.utility import mm_helper


@allure.feature("多模态对话")
@allure.story("视频输入")
@allure.title("超出最大视频数量限制")
def test_exceed_max_video_count(api_client, request):
    """测试超出模型支持的最大视频数量"""
    video_num = request.config.getoption("--videoNum")
    if video_num <= 0:
        pytest.skip("模型不支持视频输入")
    
    # 超出最大支持数量
    exceed_count = video_num + 1
    videos = mm_helper.get_random_videos(exceed_count)
    if len(videos) < exceed_count:
        video_urls = mm_helper.get_video_urls(exceed_count - len(videos))
        videos.extend(video_urls)
    
    if len(videos) < exceed_count:
        pytest.skip(f"没有足够的测试视频，需要{exceed_count}个")
    
    prompt = "请描述这些视频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, videos=videos, source_type="base64")]
    
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
@allure.story("视频输入")
@allure.title("无效的视频Base64数据")
def test_invalid_video_base64(api_client, request):
    """测试无效的视频Base64数据"""
    video_num = request.config.getoption("--videoNum")
    if video_num < 1:
        pytest.skip("模型不支持视频输入")
    
    prompt = "请描述这个视频的内容"
    # 使用无效的Base64数据
    invalid_videos = ["data:video/mp4;base64,invalid_base64_data"]
    messages = [mm_helper.build_multimodal_message(prompt, videos=invalid_videos, source_type="base64")]
    
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
@allure.story("视频输入")
@allure.title("无效的视频URL")
def test_invalid_video_url(api_client, request):
    """测试无效的视频URL"""
    video_num = request.config.getoption("--videoNum")
    if video_num < 1:
        pytest.skip("模型不支持视频输入")
    
    prompt = "请描述这个视频的内容"
    # 使用无效的URL
    invalid_videos = ["https://invalid-url/nonexistent.mp4"]
    messages = [mm_helper.build_multimodal_message(prompt, videos=invalid_videos, source_type="url")]
    
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
@allure.story("视频输入")
@allure.title("视频格式不支持")
def test_unsupported_video_format(api_client, request):
    """测试不支持的视频格式"""
    video_num = request.config.getoption("--videoNum")
    if video_num < 1:
        pytest.skip("模型不支持视频输入")
    
    prompt = "请描述这个视频的内容"
    # 使用不支持的视频格式（如.avi）
    invalid_videos = ["data:video/avi;base64,AAAAAA"]
    messages = [mm_helper.build_multimodal_message(prompt, videos=invalid_videos, source_type="base64")]
    
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
@allure.story("视频输入")
@allure.title("视频输入无prompt")
def test_video_without_prompt(api_client, request):
    """测试视频输入但没有prompt"""
    video_num = request.config.getoption("--videoNum")
    if video_num < 1:
        pytest.skip("模型不支持视频输入")
    
    videos = mm_helper.get_random_videos(1)
    if not videos:
        pytest.skip("没有可用的测试视频")
    
    # 空prompt
    messages = [mm_helper.build_multimodal_message("", videos=videos, source_type="base64")]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_not_500(response)