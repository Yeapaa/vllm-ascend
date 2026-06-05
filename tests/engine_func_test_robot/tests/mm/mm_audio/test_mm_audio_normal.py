"""多模态对话 - 音频输入 - 正常场景测试"""
import pytest
import allure
from engine_func_test_robot.utility import assertion
from engine_func_test_robot.utility import mm_helper


@allure.feature("多模态对话")
@allure.story("音频输入")
@allure.title("单音频输入-Base64")
def test_single_audio_base64(api_client, request):
    """测试单个音频输入（Base64方式）"""
    audio_num = request.config.getoption("--audioNum")
    if audio_num < 1:
        pytest.skip("模型不支持音频输入")
    
    audios = mm_helper.get_random_audios(1)
    if not audios:
        pytest.skip("没有可用的测试音频")
    
    prompt = "请描述这段音频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, audios=audios, source_type="base64")]
    
    mm_helper.attach_source_files(audios=audios, source_type="base64")
    
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
@allure.story("音频输入")
@allure.title("单音频输入-URL")
def test_single_audio_url(api_client, request):
    """测试单个音频输入（URL方式）"""
    audio_num = request.config.getoption("--audioNum")
    if audio_num < 1:
        pytest.skip("模型不支持音频输入")
    
    audio_urls = mm_helper.get_random_audio_urls(1)
    if not audio_urls:
        pytest.skip("没有可用的测试音频URL")
    
    prompt = "请描述这段音频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, audios=audio_urls, source_type="url")]
    
    mm_helper.attach_source_files(audios=audio_urls, source_type="url")
    
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
@allure.story("音频输入")
@allure.title("多音频输入-Base64")
def test_multiple_audios_base64(api_client, request):
    """测试多个音频输入（Base64方式）"""
    audio_num = request.config.getoption("--audioNum")
    if audio_num < 2:
        pytest.skip("模型不支持多音频输入")
    
    audios = mm_helper.get_random_audios(audio_num)
    if len(audios) < audio_num:
        pytest.skip(f"没有足够的测试音频，需要{audio_num}个，实际{len(audios)}个")
    
    prompt = "请描述这些音频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, audios=audios, source_type="base64")]
    
    mm_helper.attach_source_files(audios=audios, source_type="base64")
    
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
@allure.story("音频输入")
@allure.title("多音频输入-URL")
def test_multiple_audios_url(api_client, request):
    """测试多个音频输入（URL方式）"""
    audio_num = request.config.getoption("--audioNum")
    if audio_num < 2:
        pytest.skip("模型不支持多音频输入")
    
    audio_urls = mm_helper.get_audio_urls(audio_num)
    if len(audio_urls) < audio_num:
        pytest.skip(f"没有足够的测试音频URL，需要{audio_num}个，实际{len(audio_urls)}个")
    
    prompt = "请描述这些音频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, audios=audio_urls, source_type="url")]
    
    mm_helper.attach_source_files(audios=audio_urls, source_type="url")
    
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
@allure.story("音频输入")
@allure.title("音频输入流式响应")
def test_audio_input_stream(api_client, request):
    """测试音频输入流式响应"""
    audio_num = request.config.getoption("--audioNum")
    if audio_num < 1:
        pytest.skip("模型不支持音频输入")
    
    audios = mm_helper.get_random_audios(1)
    if not audios:
        pytest.skip("没有可用的测试音频")
    
    prompt = "请描述这段音频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, audios=audios, source_type="base64")]
    
    mm_helper.attach_source_files(audios=audios, source_type="base64")
    
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