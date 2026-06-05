"""图像生成接口异常场景测试用例"""
import pytest
import allure

from engine_func_test_robot.utility import assertion
from engine_func_test_robot.utility import image_helper as img_helper


@allure.epic("图像生成接口")
@allure.feature("qwen_image_gen模型")
@allure.story("异常场景")
class TestImageGenerationsAbnormal:
    """图像生成接口异常场景测试"""

    @pytest.fixture(autouse=True)
    def setup(self, api_client):
        """测试前置设置"""
        self.api_client = api_client

    @allure.title("缺少必填参数prompt")
    def test_missing_prompt(self):
        """测试缺少prompt参数"""
        data = {"size": "1024x1024"}
        img_helper.attach_generation_request(data, "请求参数-缺少prompt")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 422 Unprocessable Entity - 缺少必填字段
        assertion.assert_error_code_400(response)

    @allure.title("prompt参数为空")
    def test_empty_prompt(self):
        """测试prompt参数为空字符串"""
        data = {
            "prompt": "",
            "size": "1024x1024"
        }
        img_helper.attach_generation_request(data, "请求参数-prompt为空")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 空prompt可能返回422或400
        assertion.assert_error_code_400(response)

    @allure.title("无效的n参数-小于最小值")
    def test_invalid_n_below_min(self):
        """测试n参数小于最小值"""
        data = {
            "prompt": "a test image",
            "n": 0
        }
        img_helper.attach_generation_request(data, "请求参数-n=0")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 422 - n必须>=1
        assertion.assert_error_code_400(response)

    @allure.title("无效的n参数-大于最大值")
    def test_invalid_n_above_max(self):
        """测试n参数大于最大值"""
        data = {
            "prompt": "a test image",
            "n": 11
        }
        img_helper.attach_generation_request(data, "请求参数-n=11")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 422 - n必须<=10
        assertion.assert_error_code_400(response)

    @allure.title("无效的n参数-负数")
    def test_invalid_n_negative(self):
        """测试n参数为负数"""
        data = {
            "prompt": "a test image",
            "n": -1
        }
        img_helper.attach_generation_request(data, "请求参数-n=-1")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的n参数-非整数")
    @pytest.mark.parametrize("n", ["abc", 1.5, True, None, [], {}])
    def test_invalid_n_type(self, n):
        """测试n参数为非整数类型"""
        data = {
            "prompt": "a test image",
            "n": n
        }
        img_helper.attach_generation_request(data, f"请求参数-n={n}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的size格式")
    @pytest.mark.parametrize("size", ["invalid", "1024", "1024x", "x1024", "abcxdef", "1024x1024x1024", ""])
    def test_invalid_size_format(self, size):
        """测试无效的size格式"""
        data = {
            "prompt": "a test image",
            "size": size
        }
        img_helper.attach_generation_request(data, f"请求参数-size={size}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的size-零尺寸")
    def test_zero_size(self):
        """测试size为0"""
        data = {
            "prompt": "a test image",
            "size": "0x0"
        }
        img_helper.attach_generation_request(data, "请求参数-size=0x0")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的size-负数尺寸")
    def test_negative_size(self):
        """测试含负数的size"""
        data = {
            "prompt": "a test image",
            "size": "-100x100"
        }
        img_helper.attach_generation_request(data, "请求参数-size=-100x100")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的response_format")
    @pytest.mark.parametrize("response_format", ["url", "invalid", "json", "base64"])
    def test_invalid_response_format(self, response_format):
        """测试无效的response_format值（只支持b64_json）"""
        data = {
            "prompt": "a test image",
            "response_format": response_format
        }
        img_helper.attach_generation_request(data, f"请求参数-response_format={response_format}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 只支持b64_json格式
        assertion.assert_error_code_400(response)

    @allure.title("无效的num_inference_steps-小于最小值")
    def test_invalid_num_inference_steps_below_min(self):
        """测试num_inference_steps小于最小值"""
        data = {
            "prompt": "a test image",
            "num_inference_steps": 0
        }
        img_helper.attach_generation_request(data, "请求参数-steps=0")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的num_inference_steps-大于最大值")
    def test_invalid_num_inference_steps_above_max(self):
        """测试num_inference_steps大于最大值"""
        data = {
            "prompt": "a test image",
            "num_inference_steps": 201
        }
        img_helper.attach_generation_request(data, "请求参数-steps=201")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的num_inference_steps-负数")
    def test_invalid_num_inference_steps_negative(self):
        """测试num_inference_steps为负数"""
        data = {
            "prompt": "a test image",
            "num_inference_steps": -10
        }
        img_helper.attach_generation_request(data, "请求参数-steps=-10")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的guidance_scale-小于最小值")
    def test_invalid_guidance_scale_below_min(self):
        """测试guidance_scale小于最小值"""
        data = {
            "prompt": "a test image",
            "guidance_scale": -0.1
        }
        img_helper.attach_generation_request(data, "请求参数-guidance_scale=-0.1")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的guidance_scale-大于最大值")
    def test_invalid_guidance_scale_above_max(self):
        """测试guidance_scale大于最大值"""
        data = {
            "prompt": "a test image",
            "guidance_scale": 20.1
        }
        img_helper.attach_generation_request(data, "请求参数-guidance_scale=20.1")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的true_cfg_scale-小于最小值")
    def test_invalid_true_cfg_scale_below_min(self):
        """测试true_cfg_scale小于最小值"""
        data = {
            "prompt": "a test image",
            "true_cfg_scale": -0.1
        }
        img_helper.attach_generation_request(data, "请求参数-true_cfg_scale=-0.1")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的true_cfg_scale-大于最大值")
    def test_invalid_true_cfg_scale_above_max(self):
        """测试true_cfg_scale大于最大值"""
        data = {
            "prompt": "a test image",
            "true_cfg_scale": 20.1
        }
        img_helper.attach_generation_request(data, "请求参数-true_cfg_scale=20.1")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的layers-小于最小值")
    @pytest.mark.parametrize("layers", [0, 1, 2])
    def test_invalid_layers_below_min(self, layers):
        """测试layers小于最小值（最小为3）"""
        data = {
            "prompt": "a test image",
            "layers": layers
        }
        img_helper.attach_generation_request(data, f"请求参数-layers={layers}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的layers-大于最大值")
    @pytest.mark.parametrize("layers", [11, 12, 100])
    def test_invalid_layers_above_max(self, layers):
        """测试layers大于最大值（最大为10）"""
        data = {
            "prompt": "a test image",
            "layers": layers
        }
        img_helper.attach_generation_request(data, f"请求参数-layers={layers}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的seed类型")
    @pytest.mark.parametrize("seed", ["abc", 1.5, True, False, None, [], {}])
    def test_invalid_seed_type(self, seed):
        """测试seed为非整数类型"""
        data = {
            "prompt": "a test image",
            "seed": seed
        }
        img_helper.attach_generation_request(data, f"请求参数-seed={seed}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("超长prompt")
    def test_exceedingly_long_prompt(self):
        """测试超长的prompt"""
        data = {
            "prompt": "a" * 100000,  # 100k字符
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, "请求参数-超长prompt(100k)")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 可能返回错误或正常处理
        assertion.assert_error_code_400(response)

    @allure.title("特殊字符prompt")
    @pytest.mark.parametrize("prompt", [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "${system.prompt}",
        "../../../etc/passwd",
        "null",
        "undefined"
    ])
    def test_special_characters_prompt(self, prompt):
        """测试包含特殊字符的prompt"""
        data = {
            "prompt": prompt,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, "请求参数-特殊字符prompt")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 特殊字符可能被正常处理或报错
        assertion.assert_error_code_400(response)

    @allure.title("无效的user参数")
    def test_invalid_user(self):
        """测试空的user参数"""
        data = {
            "prompt": "a test image",
            "user": ""
        }
        img_helper.attach_generation_request(data, "请求参数-空user")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 空user可能被忽略或报错
        assertion.assert_error_code_400(response)

    @allure.title("无效的negative_prompt类型")
    @pytest.mark.parametrize("negative_prompt", [123, True, [], {}])
    def test_invalid_negative_prompt_type(self, negative_prompt):
        """测试negative_prompt为非字符串类型"""
        data = {
            "prompt": "a test image",
            "negative_prompt": negative_prompt
        }
        img_helper.attach_generation_request(data, f"请求参数-negative_prompt={negative_prompt}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的model参数")
    def test_invalid_model(self):
        """测试无效的model名称（服务端可能忽略或报错）"""
        data = {
            "prompt": "a test image",
            "model": "invalid_model_name_xyz123"
        }
        img_helper.attach_generation_request(data, "请求参数-无效model")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 服务端可能忽略model参数返回正常响应，或返回error code 400
        resp_json = response.json()
        if resp_json.get("error_code") == 400:
            assertion.assert_error_code_400(response)
        else:
            # 正常响应，校验响应体
            img_helper.assert_image_generation_response_fields(response, "无效model参数-正常响应")

    @allure.title("多参数同时无效")
    def test_multiple_invalid_parameters(self):
        """测试多个参数同时无效"""
        data = {
            "prompt": "",  # 空prompt
            "n": 100,  # n超出范围
            "size": "invalid",  # 无效size
        }
        img_helper.attach_generation_request(data, "请求参数-多参数无效")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 应该返回错误
        assertion.assert_error_code_400(response)

    @allure.title("空的请求体")
    def test_empty_request_body(self):
        """测试完全空的请求体"""
        data = {}
        img_helper.attach_generation_request(data, "请求参数-空请求体")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 缺少必填参数prompt
        assertion.assert_error_code_400(response)

    @allure.title("未知参数-应该被忽略")
    def test_unknown_parameter(self):
        """测试未知参数（应被忽略不影响正常请求）"""
        data = {
            "prompt": "a test image",
            "unknown_param": "some_value",
            "another_unknown": 123
        }
        img_helper.attach_generation_request(data, "请求参数-包含未知参数")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 未知参数可能被忽略，请求正常处理
        assertion.assert_status_code_200(response)

    @allure.title("无效的prompt参数类型")
    @pytest.mark.parametrize("prompt", [123, 1.5, True, False, None, [], {}, {"key": "value"}])
    def test_invalid_prompt_type(self, prompt):
        """测试prompt为非字符串类型"""
        data = {
            "prompt": prompt,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, f"请求参数-prompt={prompt}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 非字符串prompt应返回422校验错误
        assertion.assert_error_code_400(response)

    @allure.title("无效的size参数类型")
    @pytest.mark.parametrize("size", [123, 12.5, True, [], {}])
    def test_invalid_size_type(self, size):
        """测试size为非字符串类型"""
        data = {
            "prompt": "a test image",
            "size": size
        }
        img_helper.attach_generation_request(data, f"请求参数-size={size}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的num_inference_steps参数类型")
    @pytest.mark.parametrize("steps", ["abc", 1.5, True, False, None, [], {}])
    def test_invalid_num_inference_steps_type(self, steps):
        """测试num_inference_steps为非整数类型"""
        data = {
            "prompt": "a test image",
            "num_inference_steps": steps
        }
        img_helper.attach_generation_request(data, f"请求参数-steps={steps}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的guidance_scale参数类型")
    @pytest.mark.parametrize("guidance_scale", ["abc", True, False, None, [], {}])
    def test_invalid_guidance_scale_type(self, guidance_scale):
        """测试guidance_scale为非数值类型"""
        data = {
            "prompt": "a test image",
            "guidance_scale": guidance_scale
        }
        img_helper.attach_generation_request(data, f"请求参数-guidance_scale={guidance_scale}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的true_cfg_scale参数类型")
    @pytest.mark.parametrize("true_cfg_scale", ["abc", True, False, None, [], {}])
    def test_invalid_true_cfg_scale_type(self, true_cfg_scale):
        """测试true_cfg_scale为非数值类型"""
        data = {
            "prompt": "a test image",
            "true_cfg_scale": true_cfg_scale
        }
        img_helper.attach_generation_request(data, f"请求参数-true_cfg_scale={true_cfg_scale}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的layers参数类型")
    @pytest.mark.parametrize("layers", ["abc", 1.5, True, False, None, [], {}])
    def test_invalid_layers_type(self, layers):
        """测试layers为非整数类型"""
        data = {
            "prompt": "a test image",
            "layers": layers
        }
        img_helper.attach_generation_request(data, f"请求参数-layers={layers}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的lora参数格式")
    @pytest.mark.parametrize("lora", [
        "invalid_string",
        123,
        [],
        "null"
    ])
    def test_invalid_lora_format(self, lora):
        """测试lora参数为非对象格式"""
        data = {
            "prompt": "a test image",
            "lora": lora,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, f"请求参数-lora={lora}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("lora缺少必填字段")
    def test_lora_missing_required_fields(self):
        """测试lora参数缺少必填字段"""
        data = {
            "prompt": "a test image",
            "lora": {"name": "test"},
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, "请求参数-lora缺少path")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的vae参数类型")
    @pytest.mark.parametrize("vae_param", ["vae_use_slicing", "vae_use_tiling"])
    @pytest.mark.parametrize("value", ["yes", 123, "true", "false", [], {}])
    def test_invalid_vae_params_type(self, vae_param, value):
        """测试vae参数为非布尔类型"""
        data = {
            "prompt": "a test image",
            vae_param: value,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, f"请求参数-{vae_param}={value}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("超大size请求")
    def test_exceedingly_large_size(self):
        """测试超大尺寸请求"""
        data = {
            "prompt": "a test image",
            "size": "4096x4096"
        }
        img_helper.attach_generation_request(data, "请求参数-超大size=4096x4096")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("组合异常-多个无效参数")
    def test_combined_invalid_params(self):
        """测试多个无效参数组合"""
        data = {
            "prompt": "",
            "n": 0,
            "num_inference_steps": -1
        }
        img_helper.attach_generation_request(data, "请求参数-多参数无效组合")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("极小尺寸测试")
    def test_minimal_size(self):
        """测试极小尺寸请求"""
        data = {
            "prompt": "a test image",
            "size": "1x1"
        }
        img_helper.attach_generation_request(data, "请求参数-极小size=1x1")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的output_format参数类型")
    @pytest.mark.parametrize("output_format", [123, 1.5, True, False, None, [], {}])
    def test_invalid_output_format_type(self, output_format):
        """测试output_format为非字符串类型"""
        data = {
            "prompt": "a test image",
            "output_format": output_format
        }
        img_helper.attach_generation_request(data, f"请求参数-output_format={output_format}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的output_format值")
    @pytest.mark.parametrize("output_format", ["gif", "bmp", "tiff", "svg", "pdf", "invalid", ""])
    def test_invalid_output_format_value(self, output_format):
        """测试无效的output_format值（只支持png/jpeg/webp）"""
        data = {
            "prompt": "a test image",
            "output_format": output_format
        }
        img_helper.attach_generation_request(data, f"请求参数-output_format={output_format}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 无效的output_format可能返回错误或按默认格式处理
        resp_json = response.json()
        if resp_json.get("error_code") == 400:
            assertion.assert_error_code_400(response)
        else:
            # 正常响应，校验响应体
            img_helper.assert_image_generation_response_fields(response, f"无效output_format={output_format}-正常响应")

    @allure.title("无效的output_compression参数类型")
    @pytest.mark.parametrize("output_compression", ["abc", 1.5, True, False, None, [], {}])
    def test_invalid_output_compression_type(self, output_compression):
        """测试output_compression为非整数类型"""
        data = {
            "prompt": "a test image",
            "output_format": "jpeg",
            "output_compression": output_compression
        }
        img_helper.attach_generation_request(data, f"请求参数-output_compression={output_compression}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的output_compression值-小于最小值")
    @pytest.mark.parametrize("output_compression", [-1, -100, 0])
    def test_invalid_output_compression_below_min(self, output_compression):
        """测试output_compression小于最小值（最小为1）"""
        data = {
            "prompt": "a test image",
            "output_format": "jpeg",
            "output_compression": output_compression
        }
        img_helper.attach_generation_request(data, f"请求参数-output_compression={output_compression}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("无效的output_compression值-大于最大值")
    @pytest.mark.parametrize("output_compression", [101, 200, 1000])
    def test_invalid_output_compression_above_max(self, output_compression):
        """测试output_compression大于最大值（最大为100）"""
        data = {
            "prompt": "a test image",
            "output_format": "jpeg",
            "output_compression": output_compression
        }
        img_helper.attach_generation_request(data, f"请求参数-output_compression={output_compression}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)

    @allure.title("output_compression与png格式组合")
    def test_output_compression_with_png(self):
        """测试output_compression与png格式组合（png不支持压缩参数）"""
        data = {
            "prompt": "a test image",
            "output_format": "png",
            "output_compression": 50
        }
        img_helper.attach_generation_request(data, "请求参数-png+compression=50")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # png格式不支持压缩参数，可能被忽略或报错
        resp_json = response.json()
        if resp_json.get("error_code") == 400:
            assertion.assert_error_code_400(response)
        else:
            # 正常响应，校验响应体
            img_helper.assert_image_generation_response_fields(response, "png+compression-正常响应")

    @allure.title("无效的response_format参数类型")
    @pytest.mark.parametrize("response_format", [123, 1.5, True, False, None, [], {}])
    def test_invalid_response_format_type(self, response_format):
        """测试response_format为非字符串类型"""
        data = {
            "prompt": "a test image",
            "response_format": response_format
        }
        img_helper.attach_generation_request(data, f"请求参数-response_format={response_format}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_error_code_400(response)