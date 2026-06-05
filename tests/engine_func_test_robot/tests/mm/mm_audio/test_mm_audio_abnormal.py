"""多模态对话 - 音频输入 - 异常场景测试"""
import pytest
import allure
from engine_func_test_robot.utility import assertion
from engine_func_test_robot.utility import mm_helper


@allure.feature("多模态对话")
@allure.story("音频输入")
@allure.title("超出最大音频数量限制")
def test_exceed_max_audio_num(api_client, request):
    """测试超出模型支持的音频数量限制"""
    audio_num = request.config.getoption("--audioNum")
    if audio_num < 1:
        pytest.skip("模型不支持音频输入")
    
    # 尝试发送比限制多1个音频
    exceed_num = audio_num + 1
    audios = mm_helper.get_random_audios(exceed_num)
    if len(audios) < exceed_num:
        pytest.skip(f"没有足够的测试音频，需要{exceed_num}个")
    
    prompt = "请描述这些音频的内容"
    messages = [mm_helper.build_multimodal_message(prompt, audios=audios, source_type="base64")]
    
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
@allure.story("音频输入")
@allure.title("无效的音频Base64数据")
def test_invalid_audio_base64(api_client, request):
    """测试无效的音频Base64数据"""
    audio_num = request.config.getoption("--audioNum")
    if audio_num < 1:
        pytest.skip("模型不支持音频输入")
    
    prompt = "请描述这段音频的内容"
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "audio_url",
                "audio_url": {
                    "url": "audio/mp3;base64,invalid_base64_data"
                }
            }
        ]
    }]
    
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
@allure.story("音频输入")
@allure.title("无效的音频URL")
def test_invalid_audio_url(api_client, request):
    """测试无效的音频URL"""
    audio_num = request.config.getoption("--audioNum")
    if audio_num < 1:
        pytest.skip("模型不支持音频输入")
    
    prompt = "请描述这段音频的内容"
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "audio_url",
                "audio_url": {
                    "url": "https://invalid.example.com/audio.mp3"
                }
            }
        ]
    }]
    
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
@allure.story("音频输入")
@allure.title("音频URL格式错误")
def test_malformed_audio_url(api_client, request):
    """测试音频URL格式错误"""
    audio_num = request.config.getoption("--audioNum")
    if audio_num < 1:
        pytest.skip("模型不支持音频输入")
    
    prompt = "请描述这段音频的内容"
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "audio_url",
                "audio_url": {
                    "url": "not_a_valid_url"
                }
            }
        ]
    }]
    
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
@allure.story("音频输入")
@allure.title("音频类型不匹配")
def test_audio_type_mismatch(api_client, request):
    """测试音频文件类型与声明类型不匹配"""
    audio_num = request.config.getoption("--audioNum")
    if audio_num < 1:
        pytest.skip("模型不支持音频输入")
    
    # 使用图片文件作为音频输入
    images = mm_helper.get_random_images(1)
    if not images:
        pytest.skip("没有可用的测试图片")
    
    # 将图片转换为base64格式，但作为音频类型传入
    image_content = mm_helper.build_image_content(images[0], source_type="base64")
    image_url = image_content["image_url"]["url"]
    
    prompt = "请描述这段音频的内容"
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "audio_url",
                "audio_url": {
                    "url": image_url  # 传入图片base64而非音频
                }
            }
        ]
    }]
    
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
@allure.story("音频输入")
@allure.title("空音频数据")
def test_empty_audio_data(api_client, request):
    """测试空音频数据"""
    audio_num = request.config.getoption("--audioNum")
    if audio_num < 1:
        pytest.skip("模型不支持音频输入")
    
    prompt = "请描述这段音频的内容"
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "audio_url",
                "audio_url": {
                    "url": "audio/mp3;base64,"
                }
            }
        ]
    }]
    
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
@allure.story("音频输入")
@allure.title("不支持的音频格式")
def test_unsupported_audio_format(api_client, request):
    """测试不支持的音频格式"""
    audio_num = request.config.getoption("--audioNum")
    if audio_num < 1:
        pytest.skip("模型不支持音频输入")
    
    prompt = "请描述这段音频的内容"
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "audio_url",
                "audio_url": {
                    "url": "audio/xyz;base64,dGVzdA=="
                }
            }
        ]
    }]
    
    request_body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 500
    }
    
    mm_helper.attach_multimodal_request(request_body, "请求体")
    response = api_client.post("/v1/chat/completions", json=request_body)
    mm_helper.attach_response_body(response, "响应体")
    
    assertion.assert_error_code_400(response)


