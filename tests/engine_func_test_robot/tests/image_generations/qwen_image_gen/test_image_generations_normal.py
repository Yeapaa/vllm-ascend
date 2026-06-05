"""图像生成接口正常场景测试用例"""
import pytest
import allure

from engine_func_test_robot.utility import assertion
from engine_func_test_robot.utility import image_helper as img_helper


@allure.epic("图像生成接口")
@allure.feature("qwen_image_gen模型")
@allure.story("正常场景")
class TestImageGenerationsNormal:
    """图像生成接口正常场景测试"""

    @pytest.fixture(autouse=True)
    def setup(self, api_client):
        """测试前置设置"""
        self.api_client = api_client

    @allure.title("单图生成-基础测试")
    def test_single_image_basic(self):
        """测试单图生成基础功能"""
        data = {
            "prompt": "a beautiful sunset over the ocean",
            "size": "1024x1024"
        }
        img_helper.attach_generation_request(data, "请求参数")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # 验证响应
        assertion.assert_status_code_200(response)
        resp_json = img_helper.assert_image_generation_response_fields(response, "单图生成")
        assert len(resp_json["data"]) == 1, "默认应生成1张图片"

        # 验证图片数据
        for item in resp_json["data"]:
            if item.get("b64_json"):
                img_data = img_helper.decode_b64_image(item["b64_json"])
                width, height = img_helper.get_image_dimensions(img_data)
                assert width and height, "图片尺寸应有效"

    @allure.title("多图生成-指定n参数")
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
    def test_generate_multiple_images(self, n):
        """测试生成多张图片（n参数）"""
        data = {
            "prompt": "a cute cat playing with a ball",
            "n": n,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, f"请求参数-n={n}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = img_helper.assert_image_generation_response_fields(response, f"多图生成-n={n}")
        assert len(resp_json["data"]) == n, f"应生成{n}张图片"

        # 校验所有图片都可以解码，解码格式与响应体的输出格式一致，解码后的真实尺寸和size一致
        output_format = resp_json.get("output_format", "png")  # 默认png
        expected_size = img_helper.parse_size_string("512x512")
        for idx, item in enumerate(resp_json["data"]):
            img_data, width, height = img_helper.decode_and_validate_image(
                item["b64_json"],
                expected_format=output_format,
                expected_size=expected_size,
                msg=f"图片[{idx}]"
            )
            assert width and height, f"图片[{idx}]尺寸无效"

    @allure.title("不同尺寸测试")
    @pytest.mark.parametrize("size", [
        "256x256", "512x512", "768x768", "1024x1024",
        "512x768", "768x512", "1024x1792", "1792x1024"
    ])
    def test_different_sizes(self, size):
        """测试不同尺寸的图片生成"""
        data = {
            "prompt": "a mountain landscape",
            "size": size
        }
        img_helper.attach_generation_request(data, f"请求参数-size={size}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = img_helper.assert_image_generation_response_fields(response, f"尺寸测试-{size}")

    @allure.title("带negative_prompt测试")
    def test_with_negative_prompt(self):
        """测试带负面提示词的生成"""
        data = {
            "prompt": "a portrait of a woman",
            "negative_prompt": "blurry, low quality, distorted, ugly",
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, "请求参数")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        img_helper.assert_image_generation_response_fields(response, "带negative_prompt")

    @allure.title("指定seed测试")
    @pytest.mark.parametrize("seed", [0, 42, 12345, 2147483647, -1, -2147483648])
    def test_with_seed(self, seed):
        """测试指定随机种子（覆盖边界值）"""
        data = {
            "prompt": "a garden with flowers",
            "seed": seed,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, f"请求参数-seed={seed}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        img_helper.assert_image_generation_response_fields(response, f"指定seed-{seed}")

    @allure.title("指定num_inference_steps测试")
    @pytest.mark.parametrize("steps", [1, 20, 50, 100, 150, 200])
    def test_with_num_inference_steps(self, steps):
        """测试指定推理步数（覆盖边界值）"""
        data = {
            "prompt": "a futuristic city",
            "num_inference_steps": steps,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, f"请求参数-steps={steps}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        img_helper.assert_image_generation_response_fields(response, f"指定steps-{steps}")

    @allure.title("指定guidance_scale测试")
    @pytest.mark.parametrize("guidance_scale", [0.0, 5.0, 7.5, 10.0, 15.0, 20.0])
    def test_with_guidance_scale(self, guidance_scale):
        """测试指定引导比例（覆盖边界值）"""
        data = {
            "prompt": "a serene lake",
            "guidance_scale": guidance_scale,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, f"请求参数-guidance_scale={guidance_scale}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        img_helper.assert_image_generation_response_fields(response, f"指定guidance_scale-{guidance_scale}")

    @allure.title("指定true_cfg_scale测试")
    @pytest.mark.parametrize("true_cfg_scale", [0.0, 2.5, 5.0, 10.0, 20.0])
    def test_with_true_cfg_scale(self, true_cfg_scale):
        """测试指定True CFG比例"""
        data = {
            "prompt": "an abstract painting",
            "true_cfg_scale": true_cfg_scale,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, f"请求参数-true_cfg_scale={true_cfg_scale}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        img_helper.assert_image_generation_response_fields(response, f"指定true_cfg_scale-{true_cfg_scale}")

    @allure.title("不同prompt内容测试")
    @pytest.mark.parametrize("prompt", [
        "a cat",
        "a red sports car on a highway",
        "a peaceful Japanese garden with cherry blossoms, a small bridge over a koi pond, and traditional architecture",
        "cyberpunk city at night with neon lights and flying cars",
        "简约风格的客厅，现代家具，阳光从落地窗照入"
    ])
    def test_different_prompts(self, prompt):
        """测试不同内容的prompt"""
        data = {
            "prompt": prompt,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, f"请求参数-prompt={prompt[:20]}...")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        img_helper.assert_image_generation_response_fields(response, f"prompt测试")

    @allure.title("不同prompt长度测试")
    @pytest.mark.parametrize("length_name,length", [
        ("短prompt-10字符", 10),
        ("短prompt-50字符", 50),
        ("中prompt-100字符", 100),
        ("中prompt-500字符", 500),
        ("长prompt-1000字符", 1000),
        ("长prompt-5000字符", 5000)
    ])
    def test_different_prompt_lengths(self, length_name, length):
        """测试不同长度的prompt"""
        # 使用重复的描述性文本构造不同长度的prompt
        prompt = "A beautiful landscape with mountains and rivers, colorful flowers in the foreground. " * ((length // 80) + 1)
        prompt = prompt[:length]

        data = {
            "prompt": prompt,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, f"请求参数-{length_name}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        img_helper.assert_image_generation_response_fields(response, f"{length_name}测试")

    @allure.title("response_format测试-默认b64_json")
    def test_response_format_default(self):
        """测试默认响应格式为b64_json"""
        data = {
            "prompt": "a blue sky with white clouds",
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, "请求参数")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = response.json()
        assert len(resp_json["data"]) > 0, "应返回至少一张图片"
        assert "b64_json" in resp_json["data"][0], "应返回b64_json格式"

    @allure.title("response_format测试-显式指定b64_json")
    def test_response_format_b64_json(self):
        """测试显式指定b64_json响应格式"""
        data = {
            "prompt": "a green forest",
            "size": "512x512",
            "response_format": "b64_json"
        }
        img_helper.attach_generation_request(data, "请求参数")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = response.json()
        assert "b64_json" in resp_json["data"][0], "应返回b64_json格式"

    @allure.title("output_format测试-png格式")
    def test_output_format_png(self):
        """测试输出格式为png"""
        data = {
            "prompt": "a colorful bird",
            "size": "512x512",
            "output_format": "png"
        }
        img_helper.attach_generation_request(data, "请求参数-output_format=png")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = img_helper.assert_image_generation_response_fields(response, "output_format=png")
        assert resp_json.get("output_format") == "png", "输出格式应为png"
        # 校验图片解码格式与output_format一致
        for idx, item in enumerate(resp_json["data"]):
            img_data, width, height = img_helper.decode_and_validate_image(
                item["b64_json"],
                expected_format="png",
                msg=f"图片[{idx}]"
            )

    @allure.title("output_format测试-jpeg格式")
    def test_output_format_jpeg(self):
        """测试输出格式为jpeg"""
        data = {
            "prompt": "a sunny beach",
            "size": "512x512",
            "output_format": "jpeg"
        }
        img_helper.attach_generation_request(data, "请求参数-output_format=jpeg")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = img_helper.assert_image_generation_response_fields(response, "output_format=jpeg")
        assert resp_json.get("output_format") == "jpeg", "输出格式应为jpeg"
        # 校验图片解码格式与output_format一致
        for idx, item in enumerate(resp_json["data"]):
            img_data, width, height = img_helper.decode_and_validate_image(
                item["b64_json"],
                expected_format="jpeg",
                msg=f"图片[{idx}]"
            )

    @allure.title("output_format测试-webp格式")
    def test_output_format_webp(self):
        """测试输出格式为webp"""
        data = {
            "prompt": "a mountain landscape",
            "size": "512x512",
            "output_format": "webp"
        }
        img_helper.attach_generation_request(data, "请求参数-output_format=webp")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = img_helper.assert_image_generation_response_fields(response, "output_format=webp")
        assert resp_json.get("output_format") == "webp", "输出格式应为webp"
        # 校验图片解码格式与output_format一致
        for idx, item in enumerate(resp_json["data"]):
            img_data, width, height = img_helper.decode_and_validate_image(
                item["b64_json"],
                expected_format="webp",
                msg=f"图片[{idx}]"
            )

    @allure.title("output_compression测试-jpeg压缩质量")
    @pytest.mark.parametrize("compression", [1, 50, 100])
    def test_output_compression_jpeg(self, compression):
        """测试jpeg格式的压缩质量参数"""
        data = {
            "prompt": "a beautiful flower",
            "size": "512x512",
            "output_format": "jpeg",
            "output_compression": compression
        }
        img_helper.attach_generation_request(data, f"请求参数-output_compression={compression}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = img_helper.assert_image_generation_response_fields(response, f"output_compression={compression}")
        assert resp_json.get("output_format") == "jpeg", "输出格式应为jpeg"
        # 校验图片可解码
        for idx, item in enumerate(resp_json["data"]):
            img_data, width, height = img_helper.decode_and_validate_image(
                item["b64_json"],
                expected_format="jpeg",
                msg=f"图片[{idx}]"
            )

    @allure.title("output_compression测试-webp压缩质量")
    @pytest.mark.parametrize("compression", [1, 50, 100])
    def test_output_compression_webp(self, compression):
        """测试webp格式的压缩质量参数"""
        data = {
            "prompt": "a peaceful lake",
            "size": "512x512",
            "output_format": "webp",
            "output_compression": compression
        }
        img_helper.attach_generation_request(data, f"请求参数-output_compression={compression}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = img_helper.assert_image_generation_response_fields(response, f"output_compression={compression}")
        assert resp_json.get("output_format") == "webp", "输出格式应为webp"
        # 校验图片可解码
        for idx, item in enumerate(resp_json["data"]):
            img_data, width, height = img_helper.decode_and_validate_image(
                item["b64_json"],
                expected_format="webp",
                msg=f"图片[{idx}]"
            )

    @allure.title("output_compression默认值测试")
    def test_output_compression_default(self):
        """测试不指定output_compression时使用默认值100"""
        data = {
            "prompt": "a sunset view",
            "size": "512x512",
            "output_format": "jpeg"
        }
        img_helper.attach_generation_request(data, "请求参数-默认compression")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = img_helper.assert_image_generation_response_fields(response, "默认compression")
        # 校验图片可解码
        for idx, item in enumerate(resp_json["data"]):
            img_data, width, height = img_helper.decode_and_validate_image(
                item["b64_json"],
                expected_format="jpeg",
                msg=f"图片[{idx}]"
            )

    @allure.title("用户标识user参数测试")
    def test_with_user_parameter(self):
        """测试user参数（追踪用户）"""
        data = {
            "prompt": "a white snow mountain",
            "user": "test-user-001",
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, "请求参数")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        img_helper.assert_image_generation_response_fields(response, "user参数测试")

    @allure.title("边界测试-inf_steps边界值")
    def test_num_inference_steps_boundary(self):
        """测试num_inference_steps边界值（1和200）"""
        # steps=1
        data = {"prompt": "test steps 1", "num_inference_steps": 1, "size": "256x256"}
        response = img_helper.send_image_generation_request(self.api_client, data)
        assertion.assert_status_code_200(response)

        # steps=200
        data = {"prompt": "test steps 200", "num_inference_steps": 200, "size": "256x256"}
        response = img_helper.send_image_generation_request(self.api_client, data)
        assertion.assert_status_code_200(response)

    @allure.title("边界测试-guidance_scale边界值")
    def test_guidance_scale_boundary(self):
        """测试guidance_scale边界值（0.0和20.0）"""
        # scale=0.0
        data = {"prompt": "test scale 0", "guidance_scale": 0.0, "size": "256x256"}
        response = img_helper.send_image_generation_request(self.api_client, data)
        assertion.assert_status_code_200(response)

        # scale=20.0
        data = {"prompt": "test scale 20", "guidance_scale": 20.0, "size": "256x256"}
        response = img_helper.send_image_generation_request(self.api_client, data)
        assertion.assert_status_code_200(response)

    @allure.title("指定layers测试-分层模型")
    @pytest.mark.parametrize("layers", [3, 5, 8, 10])
    def test_with_layers(self, layers):
        """测试指定图层数（分层模型，支持范围3-10）"""
        data = {
            "prompt": "a layered artwork",
            "layers": layers,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, f"请求参数-layers={layers}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        img_helper.assert_image_generation_response_fields(response, f"指定layers-{layers}")

    @allure.title("layers边界值测试")
    def test_layers_boundary_values(self):
        """测试layers边界值（3和10）"""
        # layers=3（最小值）
        data = {"prompt": "test layers min", "layers": 3, "size": "256x256"}
        response = img_helper.send_image_generation_request(self.api_client, data)
        assertion.assert_status_code_200(response)

        # layers=10（最大值）
        data = {"prompt": "test layers max", "layers": 10, "size": "256x256"}
        response = img_helper.send_image_generation_request(self.api_client, data)
        assertion.assert_status_code_200(response)

    @allure.title("指定lora参数测试")
    def test_with_lora(self):
        """测试LoRA适配器参数"""
        # LoRA参数示例（需要实际的LoRA路径才能测试）
        data = {
            "prompt": "a styled artwork",
            "lora": {
                "name": "test_lora",
                "path": "/path/to/lora",
                "scale": 0.8
            },
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, "请求参数-lora")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # LoRA参数可能被忽略或返回错误，取决于服务端配置
        # 如果返回200，验证响应格式
        if response.status_code == 200:
            img_helper.assert_image_generation_response_fields(response, "lora参数测试")

    @allure.title("lora参数-int_id方式测试")
    def test_with_lora_int_id(self):
        """测试LoRA适配器参数（使用int_id）"""
        data = {
            "prompt": "a styled artwork",
            "lora": {
                "name": "test_lora",
                "lora_int_id": 12345,
                "lora_scale": 0.5
            },
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, "请求参数-lora_int_id")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        if response.status_code == 200:
            img_helper.assert_image_generation_response_fields(response, "lora_int_id参数测试")

    @allure.title("VAE优化参数-vae_use_slicing")
    @pytest.mark.parametrize("vae_use_slicing", [True, False])
    def test_with_vae_use_slicing(self, vae_use_slicing):
        """测试VAE切片参数"""
        data = {
            "prompt": "a test image",
            "vae_use_slicing": vae_use_slicing,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, f"请求参数-vae_use_slicing={vae_use_slicing}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        img_helper.assert_image_generation_response_fields(response, f"vae_use_slicing={vae_use_slicing}")

    @allure.title("VAE优化参数-vae_use_tiling")
    @pytest.mark.parametrize("vae_use_tiling", [True, False])
    def test_with_vae_use_tiling(self, vae_use_tiling):
        """测试VAE分块参数"""
        data = {
            "prompt": "a test image",
            "vae_use_tiling": vae_use_tiling,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, f"请求参数-vae_use_tiling={vae_use_tiling}")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        img_helper.assert_image_generation_response_fields(response, f"vae_use_tiling={vae_use_tiling}")

    @allure.title("极端尺寸测试-超大尺寸")
    def test_large_size_generation(self):
        """测试生成超大尺寸图片（性能测试）"""
        data = {
            "prompt": "a detailed panoramic view",
            "size": "1792x1024"
        }
        img_helper.attach_generation_request(data, "请求参数-大尺寸1792x1024")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = img_helper.assert_image_generation_response_fields(response, "大尺寸生成")

    @allure.title("多种参数组合压力测试")
    def test_combined_parameters_stress(self):
        """测试多种参数组合的压力测试"""
        data = {
            "prompt": "a complex scene with many details",
            "n": 5,
            "size": "512x512",
            "negative_prompt": "blurry, low quality, distorted",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "true_cfg_scale": 5.0,
            "seed": 12345
        }
        img_helper.attach_generation_request(data, "请求参数-组合压力测试")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        resp_json = img_helper.assert_image_generation_response_fields(response, "组合压力测试")
        assert len(resp_json["data"]) == 5, "应生成5张图片"

    @allure.title("多语言prompt测试")
    @pytest.mark.parametrize("prompt", [
        "A beautiful sunset",  # 英文
        "美しい桜の花",  # 日文
        "아름다운 풍경",  # 韩文
        "Un beau paysage",  # 法文
        "Eine schöne Landschaft",  # 德文
        "未来感的赛博朋克城市，霓虹灯闪烁，机器人行走"
    ])
    def test_multilingual_prompt(self, prompt):
        """测试多语言prompt"""
        data = {
            "prompt": prompt,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, f"请求参数-多语言prompt")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        img_helper.assert_image_generation_response_fields(response, "多语言prompt测试")

    @allure.title("含emoji的prompt测试")
    def test_prompt_with_emoji(self):
        """测试包含emoji的prompt"""
        data = {
            "prompt": "a beautiful sunset 🌅 with birds 🐦 flying",
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, "请求参数-含emoji")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        # emoji可能被正常处理或忽略
        assertion.assert_status_code_200(response)
        img_helper.assert_image_generation_response_fields(response, "含emoji的prompt")

    @allure.title("含特殊unicode字符的prompt测试")
    @pytest.mark.parametrize("prompt", [
        "art with symbols: ©®™",
        "math symbols: ∑∏∫",
        "arrows: →←↑↓",
        "currency: $€¥£"
    ])
    def test_special_unicode_prompt(self, prompt):
        """测试包含特殊unicode字符的prompt"""
        data = {
            "prompt": prompt,
            "size": "512x512"
        }
        img_helper.attach_generation_request(data, "请求参数-特殊unicode")
        response = img_helper.send_image_generation_request(self.api_client, data)
        img_helper.attach_response(response)

        assertion.assert_status_code_200(response)
        img_helper.assert_image_generation_response_fields(response, "特殊unicode测试")