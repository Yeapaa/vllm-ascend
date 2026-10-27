import allure

from ...utility import request_helper as helper
from ...utility.http_client import HTTPClient


@allure.feature("服务信息")
@allure.story("simple_server_info接口")
@allure.title("获取非空服务信息")
def test_simple_server_info(api_client, request):
    """single使用默认服务地址，pd使用同一IP的7002端口获取服务信息。"""
    engine_architecture = request.config.getoption("--engineArchitecture").strip()

    if engine_architecture == "pd":
        host = request.config.getoption("--modelServiceIP").strip()
        server_info_client = HTTPClient(base_url=f"http://{host}:7002")
    else:
        server_info_client = api_client

    response = server_info_client.get("/simple_server_info")
    helper.attach_response_body(response)

    response_json = response.json()
    assert isinstance(response_json, dict), \
        f"响应体应为字典，实际类型为{type(response_json).__name__}"
    assert response_json, "响应体字典不能为空"
