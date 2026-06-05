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
        required=True,
        help="设置模型服务ip，必传字段"
    )
    parser.addoption(
        "--modelServicePort",
        action="store",
        default="7001",
        help="设置模型服务port，默认端口7001"
    )
    parser.addoption(
        "--thinkTagOutput",
        action="store",
        type=str,
        required=True,
        help="设置模型服务是否需要输出think标签"
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
        default=1,
        help="设置图片数量，默认1"
    )
    parser.addoption(
        "--videoNum",
        action="store",
        type=int,
        default=1,
        help="设置视频数量，默认1"
    )
    parser.addoption(
        "--audioNum",
        action="store",
        type=int,
        default=1,
        help="设置音频数量，默认1"
    )


# def pytest_configure(config):
#     """
#     在pytest配置阶段拦截 --modelService 选项值
#     如果不是auto，则自动添加aicloud:前缀
#     """
#     original_getoption = config.getoption
#     
#     def patched_getoption(name, *args, **kwargs):
#         value = original_getoption(name, *args, **kwargs)
#         # 只对 --modelService 选项添加前缀（"auto" 严格小写）
#         if name in ("--modelService", "modelService") and value != "auto":
#             return f"aicloud:{value}"
#         return value
#     
#     # 替换getoption方法
#     config.getoption = patched_getoption
