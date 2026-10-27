import re

import pytest
from utility.http_client import HTTPClient


@pytest.fixture(scope="session")
def api_client(request):
    host = request.config.getoption("--modelServiceIP")
    port = request.config.getoption("--modelServicePort")
    yield HTTPClient(base_url=f"http://{host}:{port}")


def pytest_addoption(parser):
    parser.addoption(
        "--modelServiceIP",
        default="127.0.0.1",
        help="设置模型服务ip，必传字段"
    )
    parser.addoption(
        "--modelServicePort",
        action="store",
        default="7001",
        help="设置模型服务port，默认端口7001"
    )
    parser.addoption(
        "--prefillNodeAddress",
        action="store",
        default="",
        help="设置prefill节点地址，格式为ip:port"
    )
    parser.addoption(
        "--decodeNodeAddress",
        action="store",
        default="",
        help="设置decode节点地址，格式为ip:port"
    )
    parser.addoption(
        "--thinkTagOutput",
        action="store",
        type=str,
        default="false",
        help="设置模型服务是否需要输出think标签"
    )
    parser.addoption(
        "--mixReasoningEnableMethod",
        action="store",
        default="no",
        help="设置混合思考模型启用关闭思考的方式，默认no非混合思考，可选值：enable_thinking、thinking"
    )
    parser.addoption(
        "--engineType",
        action="store",
        default="vllm",
        help="设置引擎类型（如vllm/sglang）"
    )
    parser.addoption(
        "--engineArchitecture",
        action="store",
        default="pd",
        choices=["pd", "single"],
        help="设置引擎架构（如pd/single），default: pd"
    )
    parser.addoption(
        "--maxModelLength",
        action="store",
        default="128",
        help="设置模型最大上下文长度，单位：k（1024）"
    )
    parser.addoption(
        "--model",
        action="store",
        default="auto",
        help="实际挂载的后端模型名称，默认auto"
    )
    parser.addoption(
        "--modelService",
        action="store",
        default="auto",
        help='模型服务标识，默认auto。如果不是auto，则是aicloud:<thetaops服务名>，示例"aicloud:thetaopsModelServiceName"'
    )
    parser.addoption(
        "--imageNum",
        action="store",
        type=int,
        default=0,
        help="设置图片数量，默认0"
    )
    parser.addoption(
        "--videoNum",
        action="store",
        type=int,
        default=0,
        help="设置视频数量，默认0"
    )
    parser.addoption(
        "--audioNum",
        action="store",
        type=int,
        default=0,
        help="设置音频数量，默认0"
    )
    parser.addoption(
        "--modelCase",
        action="store",
        default="",
        help="控制模型专属case收集：以all开头时全部收集，否则包含dsv4/deepseek*v4*、glm、kimi关键字即可收集对应case"
    )


def pytest_configure(config):
    engine_type = config.getoption("--engineType", default="vllm").strip()
    if "sglang" in engine_type.lower():
        config.option.ignore.append("tests/request_id")

    engine_architecture = config.getoption("--engineArchitecture", default="pd").strip()
    if engine_architecture == "pd":
        config.option.ignore.append("tests/n")

    mix_reasoning_enable_method = config.getoption("--mixReasoningEnableMethod", default="no").strip()
    if mix_reasoning_enable_method == "no":
        config.option.ignore.append("tests/chat_template_kwargs/test_chat_template_kwargs_enable_thinking.py")
        config.option.ignore.append("tests/chat_template_kwargs/test_chat_template_kwargs_enable_thinking_stream.py")
        config.option.ignore.append("tests/chat_template_kwargs/test_chat_template_kwargs_thinking.py")
        config.option.ignore.append("tests/chat_template_kwargs/test_chat_template_kwargs_thinking_stream.py")
    elif mix_reasoning_enable_method == "thinking":
        config.option.ignore.append("tests/chat_template_kwargs/test_chat_template_kwargs_enable_thinking.py")
        config.option.ignore.append("tests/chat_template_kwargs/test_chat_template_kwargs_enable_thinking_stream.py")
    elif mix_reasoning_enable_method == "enable_thinking":
        config.option.ignore.append("tests/chat_template_kwargs/test_chat_template_kwargs_thinking.py")
        config.option.ignore.append("tests/chat_template_kwargs/test_chat_template_kwargs_thinking_stream.py")

    model_case = config.getoption("--modelCase", default="").strip().lower()
    if not model_case.startswith("all"):
        model_case_files = {
            "tests/chat_template_kwargs/test_chat_template_kwargs_deepseekv4_drop_thinking.py": (
                "dsv4" in model_case
                or bool(re.search(r"deepseek.*?v4", model_case))
            ),
            "tests/chat_template_kwargs/test_chat_template_kwargs_glm_clear_thinking.py": (
                "glm" in model_case
            ),
            "tests/chat_template_kwargs/test_chat_template_kwargs_kimi_preserve_thinking.py": (
                "kimi" in model_case
            ),
        }
        for case_file, enabled in model_case_files.items():
            if not enabled:
                config.option.ignore.append(case_file)
