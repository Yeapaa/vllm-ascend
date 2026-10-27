import json
import allure


def send_request(api_client, uri, request_body):
    """发送请求，返回响应对象"""
    return api_client.post(uri, json=request_body, headers={"Content-Type": "application/json"})


def attach_request_body(request_body, name="请求体"):
    """Allure记录请求体"""
    with allure.step(name):
        allure.attach(
            json.dumps(request_body, indent=2, ensure_ascii=False),
            name=name,
            attachment_type=allure.attachment_type.JSON,
            extension="json"
        )


def attach_response_body(response, name="响应体"):
    """Allure记录响应体"""
    response.encoding = 'utf-8'
    content_type = response.headers.get("Content-Type", "")

    with allure.step(name):
        if "application/json" in content_type:
            allure.attach(
                json.dumps(response.json(), indent=2, ensure_ascii=False),
                name=name,
                attachment_type=allure.attachment_type.JSON,
                extension="json"
            )
        else:
            allure.attach(
                response.text.encode("utf-8"),
                name=name,
                attachment_type=allure.attachment_type.TEXT
            )
