"""多模态对话 - 图片+视频+音频输入 - 异常场景测试"""
import pytest
import allure
from engine_func_test_robot.utility import assertion
from engine_func_test_robot.utility import mm_helper


@allure.feature("多模态对话")
@allure.story("图片+视频+音频输入")
@allure.title("超出最大图片数量-base64")
def test_exceed_max_images_base64(api_client, request):
    """测试超出最大图片数量（base64方式）"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    images = mm_helper.get_random_images(image_num + 1)
    videos = mm_helper.get_random_videos(1)
    audios = mm_helper.get_random_audios(1)
    if not images or not videos or not audios:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, audios=audios, source_type="base64")]
    
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
@allure.story("图片+视频+音频输入")
@allure.title("超出最大视频数量-url")
def test_exceed_max_videos_url(api_client, request):
    """测试超出最大视频数量（URL方式）"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    images = mm_helper.get_random_image_urls(1)
    videos = mm_helper.get_random_video_urls(video_num + 1)
    audios = mm_helper.get_random_audio_urls(1)
    if not images or not videos or not audios:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, audios=audios, source_type="url")]
    
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
@allure.story("图片+视频+音频输入")
@allure.title("超出最大音频数量-base64")
def test_exceed_max_audios_base64(api_client, request):
    """测试超出最大音频数量（base64方式）"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    images = mm_helper.get_random_images(1)
    videos = mm_helper.get_random_videos(1)
    audios = mm_helper.get_random_audios(audio_num + 1)
    if not images or not videos or not audios:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(prompt, images=images, videos=videos, audios=audios, source_type="base64")]
    
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
@allure.story("图片+视频+音频输入")
@allure.title("无效图片URL")
def test_invalid_image_url(api_client, request):
    """测试无效的图片URL"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    videos = mm_helper.get_random_video_urls(1)
    audios = mm_helper.get_random_audio_urls(1)
    if not videos or not audios:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(
        prompt, 
        images=["https://invalid-url.com/nonexistent.jpg"], 
        videos=videos, 
        audios=audios, 
        source_type="url"
    )]
    
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
@allure.story("图片+视频+音频输入")
@allure.title("无效视频URL")
def test_invalid_video_url(api_client, request):
    """测试无效的视频URL"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    images = mm_helper.get_random_image_urls(1)
    audios = mm_helper.get_random_audio_urls(1)
    if not images or not audios:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(
        prompt, 
        images=images, 
        videos=["https://invalid-url.com/nonexistent.mp4"], 
        audios=audios, 
        source_type="url"
    )]
    
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
@allure.story("图片+视频+音频输入")
@allure.title("无效音频URL")
def test_invalid_audio_url(api_client, request):
    """测试无效的音频URL"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    images = mm_helper.get_random_image_urls(1)
    videos = mm_helper.get_random_video_urls(1)
    if not images or not videos:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(
        prompt, 
        images=images, 
        videos=videos, 
        audios=["https://invalid-url.com/nonexistent.mp3"], 
        source_type="url"
    )]
    
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
@allure.story("图片+视频+音频输入")
@allure.title("无效base64图片数据")
def test_invalid_base64_image(api_client, request):
    """测试无效的base64图片数据"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    videos = mm_helper.get_random_videos(1)
    audios = mm_helper.get_random_audios(1)
    if not videos or not audios:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(
        prompt, 
        images=["image/png;base64,invalid_base64_data"], 
        videos=videos, 
        audios=audios, 
        source_type="base64"
    )]
    
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
@allure.story("图片+视频+音频输入")
@allure.title("无效base64视频数据")
def test_invalid_base64_video(api_client, request):
    """测试无效的base64视频数据"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    images = mm_helper.get_random_images(1)
    audios = mm_helper.get_random_audios(1)
    if not images or not audios:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(
        prompt, 
        images=images, 
        videos=["data:video/mp4;base64,invalid_base64_data"], 
        audios=audios, 
        source_type="base64"
    )]
    
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
@allure.story("图片+视频+音频输入")
@allure.title("无效base64音频数据")
def test_invalid_base64_audio(api_client, request):
    """测试无效的base64音频数据"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    images = mm_helper.get_random_images(1)
    videos = mm_helper.get_random_videos(1)
    if not images or not videos:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(
        prompt, 
        images=images, 
        videos=videos, 
        audios=["data:audio/mp3;base64,invalid_base64_data"], 
        source_type="base64"
    )]
    
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
@allure.story("图片+视频+音频输入")
@allure.title("空图片URL")
def test_empty_image_url(api_client, request):
    """测试空的图片URL"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    videos = mm_helper.get_random_video_urls(1)
    audios = mm_helper.get_random_audio_urls(1)
    if not videos or not audios:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(
        prompt, 
        images=[""], 
        videos=videos, 
        audios=audios, 
        source_type="url"
    )]
    
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
@allure.story("图片+视频+音频输入")
@allure.title("空视频URL")
def test_empty_video_url(api_client, request):
    """测试空的视频URL"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    images = mm_helper.get_random_image_urls(1)
    audios = mm_helper.get_random_audio_urls(1)
    if not images or not audios:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(
        prompt, 
        images=images, 
        videos=[""], 
        audios=audios, 
        source_type="url"
    )]
    
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
@allure.story("图片+视频+音频输入")
@allure.title("空音频URL")
def test_empty_audio_url(api_client, request):
    """测试空的音频URL"""
    image_num = request.config.getoption("--imageNum")
    video_num = request.config.getoption("--videoNum")
    audio_num = request.config.getoption("--audioNum")
    if image_num < 1 or video_num < 1 or audio_num < 1:
        pytest.skip("模型不支持图片、视频或音频输入")
    
    images = mm_helper.get_random_image_urls(1)
    videos = mm_helper.get_random_video_urls(1)
    if not images or not videos:
        pytest.skip("没有可用的测试资源")
    
    prompt = "请描述这些内容"
    messages = [mm_helper.build_multimodal_message(
        prompt, 
        images=images, 
        videos=videos, 
        audios=[""], 
        source_type="url"
    )]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)