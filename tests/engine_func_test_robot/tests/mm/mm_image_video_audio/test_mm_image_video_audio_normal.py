"""多模态对话 - 图片+视频+音频输入 - 正常场景测试"""
import pytest
import allure
from engine_func_test_robot.utility import assertion
from engine_func_test_robot.utility import mm_helper


@allure.feature("多模态对话")
@allure.story("图片+视频+音频输入")
@allure.title("单张图片+单个视频+单个音频输入-base64")
def test_single_image_video_audio_base64(api_client, request):
    """测试单张图片+单个视频+单个音频输入（base64方式）"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    images = mm_helper.get_random_images(1)
    videos = mm_helper.get_random_videos(1)
    audios = mm_helper.get_random_audios(1)
    if not images or not videos or not audios:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这张图片、这个视频和这段音频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, audios=audios, source_type="base64")]
    
    mm_helper.attach_source_files(images=images, videos=videos, audios=audios, source_type="base64")
    
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
@allure.story("图片+视频+音频输入")
@allure.title("单张图片+单个视频+单个音频输入-url")
def test_single_image_video_audio_url(api_client, request):
    """测试单张图片+单个视频+单个音频输入（URL方式）"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    images = mm_helper.get_random_image_urls(1)
    videos = mm_helper.get_random_video_urls(1)
    audios = mm_helper.get_random_audio_urls(1)
    if not images or not videos or not audios:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这张图片、这个视频和这段音频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, audios=audios, source_type="url")]
    
    mm_helper.attach_source_files(images=images, videos=videos, audios=audios, source_type="url")
    
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
@allure.story("图片+视频+音频输入")
@allure.title("最大数量图片+视频+音频输入-base64")
def test_max_images_videos_audios_base64(api_client, request):
    """测试最大数量的图片+视频+音频输入（base64方式）"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    images = mm_helper.get_random_images(image_num)
    videos = mm_helper.get_random_videos(video_num)
    audios = mm_helper.get_random_audios(audio_num)
    if len(images) < image_num or len(videos) < video_num or len(audios) < audio_num:
        pytest.skip(f"没有足够的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, audios=audios, source_type="base64")]
    
    mm_helper.attach_source_files(images=images, videos=videos, audios=audios, source_type="base64")
    
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
@allure.story("图片+视频+音频输入")
@allure.title("最大数量图片+视频+音频输入-url")
def test_max_images_videos_audios_url(api_client, request):
    """测试最大数量的图片+视频+音频输入（URL方式）"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    images = mm_helper.get_random_image_urls(image_num)
    videos = mm_helper.get_random_video_urls(video_num)
    audios = mm_helper.get_random_audio_urls(audio_num)
    if len(images) < image_num or len(videos) < video_num or len(audios) < audio_num:
        pytest.skip(f"没有足够的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, audios=audios, source_type="url")]
    
    mm_helper.attach_source_files(images=images, videos=videos, audios=audios, source_type="url")
    
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
@allure.story("图片+视频+音频输入")
@allure.title("流式响应-单张图片+单个视频+单个音频")
def test_stream_single_image_video_audio(api_client, request):
    """测试流式响应下的图片+视频+音频输入"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    images = mm_helper.get_random_image_urls(1)
    videos = mm_helper.get_random_video_urls(1)
    audios = mm_helper.get_random_audio_urls(1)
    if not images or not videos or not audios:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这张图片、这个视频和这段音频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, audios=audios, source_type="url")]
    
    mm_helper.attach_source_files(images=images, videos=videos, audios=audios, source_type="url")
    
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