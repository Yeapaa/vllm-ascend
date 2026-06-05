"""图像编辑接口异常场景测试用例"""
import os
import random
import pytest
import allure
from pathlib import Path

from engine_func_test_robot.utility import assertion
from engine_func_test_robot.utility import image_helper as img_helper


# 获取测试图片目录
IMAGES_DIR = Path(__file__).parent.parent.parent.parent / "data" / "images"


def get_test_images(count: int = 1) -> list:
    """随机获取指定数量的测试图片"""
    all_images = img_helper.get_all_test_images()
    if not all_images:
        pytest.skip("没有可用的测试图片")
    return random.sample(all_images, min(count, len(all_images)))


@allure.epic("图像编辑接口")
@allure.feature("qwen_image_edit_2511模型")
@allure.story("异常场景")
class TestImageEditAbnormal:
    """图像编辑接口异常场景测试"""

    @pytest.fixture(autouse=True)
    def setup(self, api_client):
        """测试前置设置"""
        self.api_client = api_client

    @allure.title("缺少必填参数model")
    def test_missing_model(self):
        """测试缺少model参数（两种情况：正常响应或业务错误码400）"""
        images = get_test_images(1)
        data = {
            "prompt": "将这张图片变成水彩画风格"
        }
        img_helper.attach_multipart_request(data, images, "请求参数-缺少model")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        # 检查响应体判断是正常响应还是业务错误
        resp_json = response.json()

        if "error" in resp_json or "code" in resp_json:
            # 情况1：服务端校验model字段，返回业务错误码400
            assertion.assert_error_code_400(response, "缺少model参数")
        else:
            # 情况2：服务端不校验model字段，返回正常响应
            # 校验响应体字段完整性
            assertion.assert_image_edit_response_fields(response, "缺少model参数")

            # 校验图片可解析
            for idx, item in enumerate(resp_json.get("data", [])):
                if "b64_json" in item and item["b64_json"]:
                    try:
                        img_data = img_helper.decode_b64_image(item["b64_json"])
                        width, height = img_helper.get_image_dimensions(img_data)
                        assert width and height, f"图片{idx}无法解析尺寸"
                    except Exception as e:
                        raise AssertionError(f"图片{idx}解析失败: {str(e)}")

    @allure.title("缺少必填参数prompt")
    def test_missing_prompt(self):
        """测试缺少prompt参数"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511"
        }
        img_helper.attach_multipart_request(data, images, "请求参数-缺少prompt")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("model参数为空")
    def test_empty_model(self):
        """测试model参数为空字符串"""
        images = get_test_images(1)
        data = {
            "model": "",
            "prompt": "将这张图片变成水彩画风格"
        }
        img_helper.attach_multipart_request(data, images, "请求参数-model为空")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("prompt参数为空")
    def test_empty_prompt(self):
        """测试prompt参数为空字符串"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": ""
        }
        img_helper.attach_multipart_request(data, images, "请求参数-prompt为空")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("超过2张图片输入")
    @pytest.mark.skip(reason="接口对图片数量没有限制，此场景不再适用")
    def test_exceed_max_images(self):
        """测试输入超过2张图片"""
        images = get_test_images(3) if len(img_helper.get_all_test_images()) >= 3 else get_test_images(2)

        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这些图片融合"
        }
        img_helper.attach_multipart_request(data, images, "请求参数-超过2张图片")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的size格式")
    @pytest.mark.parametrize("size", ["invalid", "1024", "1024x", "x1024", "abcxdef"])
    def test_invalid_size_format(self, size):
        """测试无效的size格式"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": size
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-size={size}")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的output_format")
    def test_invalid_output_format(self):
        """测试无效的output_format值（非法值按jpeg返回）"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "output_format": "invalid_format"
        }
        img_helper.attach_multipart_request(data, images, "请求参数-无效output_format")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        # 根据需求，非法output_format按jpeg返回，返回200
        assertion.assert_status_code_200(response)
        resp_json = response.json()
        assert resp_json.get("output_format") == "jpeg", f"无效output_format应默认返回jpeg，实际为{resp_json.get('output_format')}"

        # 校验返回的base64是有效图片
        data_list = resp_json.get("data", [])
        assert len(data_list) > 0, "应返回至少一个图片结果"
        for idx, item in enumerate(data_list):
            if "b64_json" in item and item["b64_json"]:
                try:
                    img_data = img_helper.decode_b64_image(item["b64_json"])
                    width, height = img_helper.get_image_dimensions(img_data)
                    assert width and height, f"图片{idx}无法解析尺寸，无效的图片数据"
                except Exception as e:
                    raise AssertionError(f"图片{idx}解码失败: {str(e)}")

    @allure.title("无效的output_compression")
    @pytest.mark.parametrize("compression", [-1, -100, 101, 150, 200])
    def test_invalid_output_compression(self, compression):
        """测试无效的output_compression值（正常范围：0-100）"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024",
            "output_format": "jpeg",
            "output_compression": compression
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-compression={compression}")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        # 400 Bad Request - 超出正常范围[0, 100]
        assertion.assert_error_code_400(response)

    @allure.title("无效的seed值")
    @pytest.mark.parametrize("seed", ["abc", "123abc", "1.5", "-0.5", "12.34", True, False, None, [], {}])
    def test_invalid_seed(self, seed):
        """测试无效的seed值（非整数类型）"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024",
            "seed": seed
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-seed={seed}")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        # 400 Bad Request - seed 必须是整数类型
        assertion.assert_error_code_400(response)

    @allure.title("无效的URL")
    def test_invalid_url(self):
        """测试无效的图片URL"""
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024",
            "url[]": "not_a_valid_url"
        }
        img_helper.attach_multipart_request(data, None, "请求参数-无效URL")
        response = img_helper.send_image_edit_request(self.api_client, data, None)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("同时提供image和url")
    def test_both_image_and_url(self):
        """测试同时提供image文件和url参数无效值"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024",
            "url[]": "https://example.com/test.png"
        }
        img_helper.attach_multipart_request(data, images, "请求参数-同时提供image和url")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("空字符串URL")
    def test_empty_url(self):
        """测试空字符串的URL"""
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024",
            "url[]": ""
        }
        img_helper.attach_multipart_request(data, None, "请求参数-空URL")
        response = img_helper.send_image_edit_request(self.api_client, data, None)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("URL不存在-404")
    def test_url_not_found(self):
        """测试访问不存在的URL（404）"""
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024",
            "url[]": "https://httpbin.org/status/404"
        }
        img_helper.attach_multipart_request(data, None, "请求参数-URL不存在")
        response = img_helper.send_image_edit_request(self.api_client, data, None)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("非图片URL")
    @pytest.mark.parametrize("url", [
        "https://httpbin.org/html",
        "https://httpbin.org/json",
        "https://httpbin.org/xml"
    ])
    def test_non_image_url(self, url):
        """测试非图片类型的URL（HTML、JSON、XML等）"""
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024",
            "url[]": url
        }
        img_helper.attach_multipart_request(data, None, f"请求参数-非图片URL({url.split('/')[-1]})")
        response = img_helper.send_image_edit_request(self.api_client, data, None)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("非HTTP协议URL")
    @pytest.mark.parametrize("url", [
        "ftp://example.com/image.jpg",
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "file:///path/to/image.jpg"
    ])
    def test_unsupported_protocol_url(self, url):
        """测试不支持的协议URL（ftp、data、file等）"""
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024",
            "url[]": url
        }
        # 从URL中提取协议名称用于日志
        protocol = url.split(":")[0]
        img_helper.attach_multipart_request(data, None, f"请求参数-非HTTP协议({protocol})")
        response = img_helper.send_image_edit_request(self.api_client, data, None)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("超长URL")
    def test_oversized_url(self):
        """测试超长URL字符串"""
        # 构造一个超长URL（超过2000字符）
        long_url = "https://example.com/" + "a" * 2000 + ".jpg"
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024",
            "url[]": long_url
        }
        img_helper.attach_multipart_request(data, None, "请求参数-超长URL")
        response = img_helper.send_image_edit_request(self.api_client, data, None)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("超大size请求")
    @pytest.mark.parametrize("size", ["2049x2048", "2048x2049", "3000x2000", "4096x1025", "4096x4096"])
    def test_oversized_size(self, size):
        """测试请求超过像素限制(4194304)的size"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": size
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-超大size={size}")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        # 超过像素限制应返回400错误
        assertion.assert_error_code_400(response)

    @allure.title("缺少图片输入")
    def test_no_image_input(self):
        """测试既没有image文件也没有url参数"""
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格"
        }
        img_helper.attach_multipart_request(data, None, "请求参数-无图片输入")
        response = img_helper.send_image_edit_request(self.api_client, data, None)
        img_helper.attach_response(response)

        # 422 Unprocessable Entity - 数据验证失败（缺少必填的图片输入）
        assertion.assert_error_code_422(response)

    @allure.title("无效的response_format")
    def test_invalid_response_format(self):
        """测试无效的response_format值"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "response_format": "invalid_format"
        }
        img_helper.attach_multipart_request(data, images, "请求参数-无效response_format")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("超长prompt")
    def test_exceedingly_long_prompt(self):
        """测试超长的prompt"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "a" * 10000  # 超长prompt
        }
        img_helper.attach_multipart_request(data, images, "请求参数-超长prompt")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("特殊字符prompt")
    @pytest.mark.parametrize("prompt", ["<script>alert('xss')</script>", "'; DROP TABLE users; --", "${system.prompt}"])
    def test_special_characters_prompt(self, prompt):
        """测试包含特殊字符的prompt"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": prompt
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-特殊字符prompt")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的negative_prompt")
    def test_empty_negative_prompt(self):
        """测试空的negative_prompt"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "negative_prompt": ""
        }
        img_helper.attach_multipart_request(data, images, "请求参数-空negative_prompt")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("不支持的图片格式")
    def test_unsupported_image_format(self, tmp_path):
        """测试不支持的图片格式"""
        # 创建一个假的图片文件
        fake_image = tmp_path / "test.txt"
        fake_image.write_text("This is not an image")

        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格"
        }
        img_helper.attach_multipart_request(data, [str(fake_image)], "请求参数-不支持的图片格式")
        response = img_helper.send_image_edit_request(self.api_client, data, [str(fake_image)])
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的user参数")
    def test_invalid_user(self):
        """测试无效的user参数"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "user": ""
        }
        img_helper.attach_multipart_request(data, images, "请求参数-无效user")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("尺寸为0")
    def test_zero_size(self):
        """测试size为0"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "0x0"
        }
        img_helper.attach_multipart_request(data, images, "请求参数-size为0")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("负数尺寸")
    def test_negative_size(self):
        """测试负数尺寸"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "-100x-100"
        }
        img_helper.attach_multipart_request(data, images, "请求参数-负数尺寸")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)