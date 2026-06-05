"""多模态对话 - 视频输入 - 正常场景测试"""
import pytest
import allure
from engine_func_test_robot.utility import assertion
from engine_func_test_robot.utility import mm_helper


@allure.feature("多模态对话")
@allure.story("视频输入")
@allure.title("单视频输入-Base64")
def test_single_video_base64(api_client, request):
    """测试单个视频输入（Base64方式）"""
    video_num = request.config.getoption("--videoNum")
    if video_num < 1:
        pytest.skip("模型不支持视频输入")
    
    videos = mm_helper.get_random_videos(1)
    if not videos:
        pytest.skip("没有可用的测试视频")
    
    prompt = "请描述这个视频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, videos=videos, source_type="base64")]
    
    mm_helper.attach_source_files(videos=videos, source_type="base64")
    
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
@allure.story("视频输入")
@allure.title("单视频输入-URL")
def test_single_video_url(api_client, request):
    """测试单个视频输入（URL方式）"""
    video_num = request.config.getoption("--videoNum")
    if video_num < 1:
        pytest.skip("模型不支持视频输入")
    
    video_urls = mm_helper.get_random_video_urls(1)
    if not video_urls:
        pytest.skip("没有可用的测试视频URL")
    
    prompt = "请描述这个视频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, videos=video_urls, source_type="url")]
    
    mm_helper.attach_source_files(videos=video_urls, source_type="url")
    
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
@allure.story("视频输入")
@allure.title("多视频输入-Base64")
def test_multiple_videos_base64(api_client, request):
    """测试多个视频输入（Base64方式）"""
    video_num = request.config.getoption("--videoNum")
    if video_num < 2:
        pytest.skip("模型不支持多视频输入")
    
    videos = mm_helper.get_random_videos(video_num)
    if len(videos) < video_num:
        pytest.skip(f"没有足够的测试视频，需要{video_num}个，实际{len(videos)}个")
    
    prompt = "请描述这些视频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, videos=videos, source_type="base64")]
    
    mm_helper.attach_source_files(videos=videos, source_type="base64")
    
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
@allure.story("视频输入")
@allure.title("多视频输入-URL")
def test_multiple_videos_url(api_client, request):
    """测试多个视频输入（URL方式）"""
    video_num = request.config.getoption("--videoNum")
    if video_num < 2:
        pytest.skip("模型不支持多视频输入")
    
    video_urls = mm_helper.get_video_urls(video_num)
    if len(video_urls) < video_num:
        pytest.skip(f"没有足够的测试视频URL，需要{video_num}个，实际{len(video_urls)}个")
    
    prompt = "请描述这些视频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, videos=video_urls, source_type="url")]
    
    mm_helper.attach_source_files(videos=video_urls, source_type="url")
    
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
@allure.story("视频输入")
@allure.title("视频输入流式响应")
def test_video_input_stream(api_client, request):
    """测试视频输入流式响应"""
    video_num = request.config.getoption("--videoNum")
    if video_num < 1:
        pytest.skip("模型不支持视频输入")
    
    videos = mm_helper.get_random_videos(1)
    if not videos:
        pytest.skip("没有可用的测试视频")
    
    prompt = "请描述这个视频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, videos=videos, source_type="base64")]
    
    mm_helper.attach_source_files(videos=videos, source_type="base64")
    
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