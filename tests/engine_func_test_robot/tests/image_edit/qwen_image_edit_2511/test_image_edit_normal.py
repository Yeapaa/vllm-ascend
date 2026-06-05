"""图像编辑接口正常场景测试用例"""
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
    """随机获取指定数量的测试图片（支持复用）"""
    all_images = img_helper.get_all_test_images()
    if not all_images:
        return []

    # 如果图片不足，通过复用现有图片达到所需数量
    if len(all_images) >= count:
        return random.sample(all_images, count)
    else:
        return (all_images * ((count // len(all_images)) + 1))[:count]


@allure.epic("图像编辑接口")
@allure.feature("qwen_image_edit_2511模型")
@allure.story("正常场景")
class TestImageEditNormal:
    """图像编辑接口正常场景测试"""

    @pytest.fixture(autouse=True)
    def setup(self, api_client):
        """测试前置设置"""
        self.api_client = api_client

    @allure.title("单图编辑-基础测试")
    def test_single_image_basic(self):
        """测试单图编辑基础功能"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024"
        }
        img_helper.attach_multipart_request(data, images, "请求参数")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        # 验证响应
        assertion.assert_status_code_200(response)
        resp_json = response.json()
        assert "data" in resp_json, "响应应包含data字段"
        assert len(resp_json["data"]) > 0, "data应包含至少一个结果"

        # 验证输出图片
        for item in resp_json["data"]:
            assert "b64_json" in item, "应包含b64_json字段"
            # 解码验证图片尺寸
            if item["b64_json"]:
                img_data = img_helper.decode_b64_image(item["b64_json"])
                width, height = img_helper.get_image_dimensions(img_data)
                resp_width, resp_height = img_helper.parse_size_string(resp_json.get("size", ""))
                if width and resp_width:
                    assert width == resp_width, f"图片实际宽度{width}与响应宽度{resp_width}不一致"
                    assert height == resp_height, f"图片实际高度{height}与响应高度{resp_height}不一致"

    @allure.title("单图编辑-指定size")
    @pytest.mark.parametrize("size", ["1024x1024", "512x512", "768x1024", "2048x2048", "2048x2047", "2047x2048"])
    def test_single_image_with_size(self, size):
        """测试单图编辑指定输出尺寸（包含像素限制边界测试）"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": size
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-size={size}")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = response.json()

        # 验证输出尺寸
        resp_size = resp_json.get("size", "")
        resp_width, resp_height = img_helper.parse_size_string(resp_size)
        req_width, req_height = img_helper.parse_size_string(size)

        if resp_width and req_width:
            # 验证尺寸向上取整到16的倍数
            expected_width = ((req_width + 15) // 16) * 16
            expected_height = ((req_height + 15) // 16) * 16
            assert resp_width == expected_width, f"宽度应为{expected_width}，实际为{resp_width}"
            assert resp_height == expected_height, f"高度应为{expected_height}，实际为{resp_height}"

    @allure.title("单图编辑-指定output_format")
    @pytest.mark.parametrize("output_format", ["png", "jpeg", "webp"])
    def test_single_image_with_format(self, output_format):
        """测试单图编辑指定输出格式"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024",
            "output_format": output_format
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-format={output_format}")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = response.json()
        assert resp_json.get("output_format") == output_format, f"输出格式应为{output_format}"

    @allure.title("单图编辑-指定seed")
    @pytest.mark.parametrize("seed", [0, 12345, -1, -999999, 999999999, 2147483647, -2147483648])
    def test_single_image_with_seed(self, seed):
        """测试单图编辑指定随机种子（覆盖边界值和极值）"""
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

        # 校验响应字段完整性
        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, f"单图编辑-seed={seed}")

    @allure.title("双图编辑-基础测试")
    def test_dual_image_basic(self):
        """测试双图编辑基础功能"""
        images = get_test_images(2)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这两张图片融合在一起",
            "size": "1024x1024"
        }
        img_helper.attach_multipart_request(data, images, "请求参数")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        # 校验响应字段完整性
        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, "双图编辑-基础测试")

    @allure.title("双图编辑-指定size和format")
    def test_dual_image_with_options(self):
        """测试双图编辑指定尺寸和格式"""
        images = get_test_images(2)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这两张图片融合成一幅画",
            "size": "1024x1024",
            "output_format": "jpeg"
        }
        img_helper.attach_multipart_request(data, images, "请求参数")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        # 校验响应字段完整性
        assertion.assert_status_code_200(response)
        resp_json = assertion.assert_image_edit_response_fields(response, "双图编辑-指定参数")
        # 校验输出格式
        assert resp_json.get("output_format") == "jpeg", f"输出格式应为jpeg，实际为{resp_json.get('output_format')}"

    @allure.title("多图编辑-5张图片输入")
    def test_multiple_images_5(self):
        """测试5张图片同时输入"""
        images = get_test_images(5)

        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这些图片融合成一幅画",
            "size": "1024x1024"
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-{len(images)}张图片")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, f"多图编辑-{len(images)}张图片")

    @allure.title("多图编辑-10张图片输入")
    def test_multiple_images_10(self):
        """测试10张图片同时输入"""
        images = get_test_images(10)

        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这些图片融合成一幅画",
            "size": "1024x1024"
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-{len(images)}张图片")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, f"多图编辑-{len(images)}张图片")

    @allure.title("多个URL输入-5个URL")
    def test_multiple_urls_5(self):
        """测试5个URL同时输入"""
        test_urls = img_helper.get_random_test_image_urls(5)

        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这些图片融合成一幅画",
            "size": "1024x1024",
            "url[]": test_urls  # 传递URL列表
        }
        img_helper.attach_multipart_request(data, None, f"请求参数-{len(test_urls)}个URL")
        response = img_helper.send_image_edit_request(self.api_client, data, None)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, f"多URL编辑-{len(test_urls)}个URL")

    @allure.title("多个URL输入-10个URL")
    def test_multiple_urls_10(self):
        """测试10个URL同时输入"""
        test_urls = img_helper.get_random_test_image_urls(10)

        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这些图片融合成一幅画",
            "size": "1024x1024",
            "url[]": test_urls
        }
        img_helper.attach_multipart_request(data, None, f"请求参数-{len(test_urls)}个URL")
        response = img_helper.send_image_edit_request(self.api_client, data, None)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, f"多URL编辑-{len(test_urls)}个URL")

    @allure.title("图片和URL组合输入-3+3")
    def test_image_url_combination_3_3(self):
        """测试3个图片文件+3个URL组合输入"""
        images = get_test_images(3) if len(img_helper.get_all_test_images()) >= 3 else get_test_images(2)
        test_urls = img_helper.get_random_test_image_urls(3)

        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这些图片融合成一幅画",
            "size": "1024x1024",
            "url[]": test_urls
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-{len(images)}张图片+{len(test_urls)}个URL")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, f"组合编辑-{len(images)}+{len(test_urls)}")

    @allure.title("图片和URL组合输入-5+5")
    def test_image_url_combination_5_5(self):
        """测试5个图片文件+5个URL组合输入"""
        images = get_test_images(5) if len(img_helper.get_all_test_images()) >= 5 else get_test_images(2)
        test_urls = img_helper.get_random_test_image_urls(5)

        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这些图片融合成一幅画",
            "size": "1024x1024",
            "url[]": test_urls
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-{len(images)}张图片+{len(test_urls)}个URL")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, f"组合编辑-{len(images)}+{len(test_urls)}")

    @allure.title("大量图片输入-20张")
    def test_large_images_input_20(self):
        """测试20张图片同时输入（压力测试）"""
        all_images = img_helper.get_all_test_images()

        # 如果图片不足20张，复用现有图片
        if len(all_images) >= 20:
            images = get_test_images(20)
        else:
            images = (all_images * ((20 // len(all_images)) + 1))[:20]

        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这些图片融合成一幅画",
            "size": "1024x1024"
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-{len(images)}张图片")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, f"大量图片编辑-{len(images)}张")

    @allure.title("极限压力测试-1000张图片输入")
    def test_extreme_images_input_1000(self):
        """测试1000张图片同时输入（极限压力测试）"""
        all_images = img_helper.get_all_test_images()

        # 如果图片不足1000张，循环复用现有图片
        if len(all_images) >= 1000:
            images = get_test_images(1000)
        else:
            # 通过循环复用现有图片达到1000张
            images = (all_images * ((1000 // len(all_images)) + 1))[:1000]

        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这些图片融合成一幅画",
            "size": "1024x1024"
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-{len(images)}张图片-极限压力测试")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, f"极限图片编辑-{len(images)}张")

    @allure.title("单图编辑-使用URL输入")
    def test_single_image_with_url(self):
        """测试使用URL输入图片"""
        # 从 image_urls 文件中随机选择一个 URL
        test_url = img_helper.get_random_test_image_url()

        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024",
            "url[]": test_url
        }
        img_helper.attach_multipart_request(data, None, f"请求参数-URL模式(url[]={test_url})")
        response = img_helper.send_image_edit_request(self.api_client, data, None)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, "单图编辑-URL模式")

    @allure.title("单图编辑-带negative_prompt")
    def test_single_image_with_negative_prompt(self):
        """测试带负面提示词的单图编辑"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024",
            "negative_prompt": "不要添加文字，不要改变主体"
        }
        img_helper.attach_multipart_request(data, images, "请求参数")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        # 校验响应字段完整性
        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, "单图编辑-negative_prompt")

    @allure.title("单图编辑-带output_compression")
    @pytest.mark.parametrize("compression", [0, 50, 100])
    def test_single_image_with_compression(self, compression):
        """测试带压缩参数的单图编辑（边界值测试：0-100）"""
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

        # 校验响应字段完整性
        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, f"单图编辑-compression={compression}")

    @allure.title("单图编辑-带response_format")
    def test_single_image_with_response_format(self):
        """测试指定响应格式"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024",
            "response_format": "b64_json"
        }
        img_helper.attach_multipart_request(data, images, "请求参数")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        # 校验响应字段完整性
        assertion.assert_status_code_200(response)
        resp_json = assertion.assert_image_edit_response_fields(response, "单图编辑-response_format")
        # 验证返回b64_json格式
        for item in resp_json.get("data", []):
            assert "b64_json" in item, "应返回b64_json格式"

    @allure.title("验证响应字段完整性")
    def test_response_fields(self):
        """验证响应包含所有必要字段"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": "将这张图片变成水彩画风格",
            "size": "1024x1024"
        }
        img_helper.attach_multipart_request(data, images, "请求参数")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = response.json()

        # 验证顶层字段
        assert "created" in resp_json, "响应应包含created字段"
        assert "data" in resp_json, "响应应包含data字段"
        assert "output_format" in resp_json, "响应应包含output_format字段"
        assert "size" in resp_json, "响应应包含size字段"

        # 验证data数组元素字段
        for item in resp_json["data"]:
            assert "b64_json" in item or "url" in item, "应包含b64_json或url字段"
            # revised_prompt可以为null
            assert "revised_prompt" in item, "应包含revised_prompt字段"

    @allure.title("不同prompt测试-内容变体")
    @pytest.mark.parametrize("prompt", [
        "将这张图片变成水彩画风格",
        "给这张图片添加一个太阳",
        "改变图片的色调为暖色",
        "将背景替换为森林"
    ])
    def test_different_prompt_contents(self, prompt):
        """测试不同的提示词内容"""
        images = get_test_images(1)
        data = {
            "model": "qwen_image_edit_2511",
            "prompt": prompt,
            "size": "1024x1024"
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-prompt={prompt[:20]}")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        # 校验响应字段完整性
        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, f"不同prompt测试-{prompt[:10]}")

    @allure.title("不同prompt测试-长度变体")
    @pytest.mark.parametrize("length_name,length", [
        ("短prompt-10字符", 10),
        ("短prompt-50字符", 50),
        ("中prompt-100字符", 100),
        ("中prompt-500字符", 500),
        ("长prompt-1000字符", 1000),
        ("长prompt-2000字符", 2000),
        ("很长prompt-5000字符", 5000),
        ("极长prompt-10000字符", 10000),
        ("超长prompt-100000字符", 100000)
    ])
    def test_different_prompt_lengths(self, length_name, length):
        """测试不同长度的提示词（覆盖边界值和极值）"""
        images = get_test_images(1)
        # 使用重复的描述性文本构造不同长度的prompt
        prompt = "请将这张图片转换成艺术风格，添加美丽的色彩和细节。" * ((length // 28) + 1)
        prompt = prompt[:length] if length > 0 else ""

        data = {
            "model": "qwen_image_edit_2511",
            "prompt": prompt,
            "size": "1024x1024"
        }
        img_helper.attach_multipart_request(data, images, f"请求参数-{length_name}")
        response = img_helper.send_image_edit_request(self.api_client, data, images)
        img_helper.attach_response(response)

        # 校验响应字段完整性
        assertion.assert_status_code_200(response)
        assertion.assert_image_edit_response_fields(response, f"{length_name}测试")