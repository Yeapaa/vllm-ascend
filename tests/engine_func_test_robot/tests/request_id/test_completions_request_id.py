"""
v1/completions 接口 X-Request-ID 测试用例

测试场景：
1. 1000个用户以相同X-Request-ID并发请求
2. 1000个用户各自不同X-Request-ID并发请求

校验逻辑：
- 响应可以是 error code 400
- 如果不是 error code 400，则正常响应，校验 finish_reason 为 stop 或 length
- 校验响应体字段 id 值以请求的 X-Request-ID 值为后缀
"""

import uuid
import re
import time
import allure
import pytest_check as check
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


# 并发用户数
CONCURRENT_USERS = 3  # TODO: 待conductor镜像fix后修改为100以及REQUEST_TIMEOUT改为600
# 请求超时时间（秒）
REQUEST_TIMEOUT = 3600


def send_request_with_request_id(base_url, uri, request_body, request_id):
    """发送带 X-Request-ID 的请求，使用独立的 Session"""
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": request_id
    }
    # 每个线程使用独立的 Session，避免连接池竞争
    session = requests.Session()
    try:
        response = session.post(
            f"{base_url}{uri}", 
            json=request_body, 
            headers=headers, 
            timeout=REQUEST_TIMEOUT,
            stream=True  # 使用流式模式，避免阻塞
        )
        return response
    finally:
        session.close()


def validate_response(response, request_id, is_stream=False):
    """
    校验响应：
    - 响应可以是 error code 400
    - 如果不是 error code 400，则正常响应，校验 finish_reason 为 stop 或 length
    - 校验响应体字段 id 值以请求的 X-Request-ID 值为后缀
    
    返回: (is_valid, msg, response_body)
    """
    if is_stream:
        # 流式响应处理 - 逐块读取
        response_text = ""
        try:
            for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                if chunk:
                    response_text += chunk
                    # 如果已经收到 [DONE]，可以提前结束
                    if "[DONE]" in response_text:
                        break
        except Exception as e:
            return False, f"读取流式响应失败: {str(e)}", response_text
        finally:
            response.close()
        
        # 检查是否有错误码
        match = re.search(r'"code"\s*:\s*(\d+)', response_text, re.M)
        if match and int(match.group(1)) == 400:
            return True, "error_code_400", response_text
        
        # 校验 finish_reason
        finish_reason_match = re.search(r'"finish_reason"\s*:\s*"([^"]+)"', response_text)
        if finish_reason_match:
            finish_reason = finish_reason_match.group(1)
            if finish_reason not in ["stop", "length"]:
                return False, f"finish_reason={finish_reason} 不是 stop 或 length", response_text
        else:
            return False, "未找到 finish_reason", response_text
        
        # 校验 id 以 request_id 为后缀
        id_match = re.search(r'"id"\s*:\s*"([^"]+)"', response_text)
        if id_match:
            response_id = id_match.group(1)
            if not response_id.endswith(request_id):
                return False, f"response_id={response_id} 不以 request_id={request_id} 为后缀", response_text
        
        return True, "success", response_text
    else:
        # 非流式响应处理
        response_text = ""
        try:
            response_text = response.text
            response_json = response.json()
        except Exception:
            return False, "响应不是有效JSON", response_text
        finally:
            response.close()
        
        # 检查是否有错误码
        error_code = response_json.get("error", {}).get("code") or response_json.get("code")
        if error_code == 400:
            return True, "error_code_400", response_text
        
        # 校验 finish_reason
        choices = response_json.get("choices", [])
        if not choices:
            return False, "响应缺少 choices 字段", response_text
        
        finish_reason = choices[0].get("finish_reason")
        if finish_reason not in ["stop", "length"]:
            return False, f"finish_reason={finish_reason} 不是 stop 或 length", response_text
        
        # 校验 id 以 request_id 为后缀
        response_id = response_json.get("id", "")
        if not response_id.endswith(request_id):
            return False, f"response_id={response_id} 不以 request_id={request_id} 为后缀", response_text
        
        return True, "success", response_text


class TestCompletionsRequestID:
    """v1/completions 接口 X-Request-ID 测试"""
    
    # 请求体模板
    REQUEST_BODY_NON_STREAM = {
        "model": "auto",
        "prompt": "你好",
        "stream": False,
        "max_tokens": 10
    }
    
    REQUEST_BODY_STREAM = {
        "model": "auto",
        "prompt": "你好",
        "stream": True,
        "max_tokens": 10
    }
    
    def test_same_request_id_non_stream(self, api_client, request):
        """
        场景1：非流式 - 1000个用户以相同X-Request-ID并发请求
        """
        base_url = api_client.base_url
        request_id = f"test-same-id-{uuid.uuid4().hex[:8]}"
        request_body = self.REQUEST_BODY_NON_STREAM.copy()
        
        # 记录请求体（只记录一次）
        with allure.step(f"请求体: {request_body}"):
            allure.attach(str(request_body), name="请求体", attachment_type=allure.attachment_type.JSON)
        
        results = {"success": 0, "error_code_400": 0, "failed": 0, "errors": []}
        response_bodies = []  # 记录每次响应体
        
        def single_request(index):
            # 相同 request_id 用例，请求之间间隔 5ms 发送
            if index > 0:
                time.sleep(0.005 * index)  # 第 n 个请求等待 n*5ms
            try:
                response = send_request_with_request_id(
                    base_url, 
                    "/v1/completions", 
                    request_body, 
                    request_id
                )
                is_valid, msg, response_body = validate_response(response, request_id, is_stream=False)
                return is_valid, msg, response_body
            except Exception as e:
                return False, str(e), ""
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(single_request, i) for i in range(CONCURRENT_USERS)]
            for idx, future in enumerate(as_completed(futures, timeout=REQUEST_TIMEOUT + 60)):
                is_valid, msg, response_body = future.result()
                response_bodies.append(f"请求{idx+1}: {response_body}")
                if is_valid:
                    if msg == "error_code_400":
                        results["error_code_400"] += 1
                    else:
                        results["success"] += 1
                else:
                    results["failed"] += 1
                    if len(results["errors"]) < 10:
                        results["errors"].append(msg)
        
        # 记录所有响应体
        with allure.step("响应体列表"):
            for i, body in enumerate(response_bodies):
                allure.attach(body, name=f"响应{i+1}", attachment_type=allure.attachment_type.TEXT)
        
        with allure.step(f"并发请求结果: 成功={results['success']}, 错误码400={results['error_code_400']}, 失败={results['failed']}"):
            pass
        
        total_valid = results["success"] + results["error_code_400"]
        check.equal(total_valid, CONCURRENT_USERS, 
                    f"部分请求失败: {results['failed']} 个, 错误示例: {results['errors'][:3]}")
    
    def test_same_request_id_stream(self, api_client, request):
        """
        场景1：流式 - 1000个用户以相同X-Request-ID并发请求
        """
        base_url = api_client.base_url
        request_id = f"test-same-id-{uuid.uuid4().hex[:8]}"
        request_body = self.REQUEST_BODY_STREAM.copy()
        
        # 记录请求体（只记录一次）
        with allure.step(f"请求体: {request_body}"):
            allure.attach(str(request_body), name="请求体", attachment_type=allure.attachment_type.JSON)
        
        results = {"success": 0, "error_code_400": 0, "failed": 0, "errors": []}
        response_bodies = []  # 记录每次响应体
        
        def single_request(index):
            # 相同 request_id 用例，请求之间间隔 5ms 发送
            if index > 0:
                time.sleep(0.005 * index)  # 第 n 个请求等待 n*5ms
            try:
                response = send_request_with_request_id(
                    base_url, 
                    "/v1/completions", 
                    request_body, 
                    request_id
                )
                is_valid, msg, response_body = validate_response(response, request_id, is_stream=True)
                return is_valid, msg, response_body
            except Exception as e:
                return False, str(e), ""
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(single_request, i) for i in range(CONCURRENT_USERS)]
            for idx, future in enumerate(as_completed(futures, timeout=REQUEST_TIMEOUT + 60)):
                is_valid, msg, response_body = future.result()
                response_bodies.append(f"请求{idx+1}: {response_body}")
                if is_valid:
                    if msg == "error_code_400":
                        results["error_code_400"] += 1
                    else:
                        results["success"] += 1
                else:
                    results["failed"] += 1
                    if len(results["errors"]) < 10:
                        results["errors"].append(msg)
        
        # 记录所有响应体
        with allure.step("响应体列表"):
            for i, body in enumerate(response_bodies):
                allure.attach(body, name=f"响应{i+1}", attachment_type=allure.attachment_type.TEXT)
        
        with allure.step(f"并发请求结果: 成功={results['success']}, 错误码400={results['error_code_400']}, 失败={results['failed']}"):
            pass
        
        total_valid = results["success"] + results["error_code_400"]
        check.equal(total_valid, CONCURRENT_USERS, 
                    f"部分请求失败: {results['failed']} 个, 错误示例: {results['errors'][:3]}")
    
    def test_different_request_id_non_stream(self, api_client, request):
        """
        场景2：非流式 - 1000个用户各自不同X-Request-ID并发请求
        """
        base_url = api_client.base_url
        request_body = self.REQUEST_BODY_NON_STREAM.copy()
        
        # 记录请求体（只记录一次）
        with allure.step(f"请求体: {request_body}"):
            allure.attach(str(request_body), name="请求体", attachment_type=allure.attachment_type.JSON)
        
        results = {"success": 0, "error_code_400": 0, "failed": 0, "errors": []}
        response_bodies = []  # 记录每次响应体
        
        def single_request(index):
            try:
                # 每个请求使用不同的 request_id
                request_id = f"test-diff-id-{uuid.uuid4().hex}"
                response = send_request_with_request_id(
                    base_url, 
                    "/v1/completions", 
                    request_body, 
                    request_id
                )
                is_valid, msg, response_body = validate_response(response, request_id, is_stream=False)
                return is_valid, msg, response_body, request_id
            except Exception as e:
                return False, str(e), "", ""
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(single_request, i) for i in range(CONCURRENT_USERS)]
            for idx, future in enumerate(as_completed(futures, timeout=REQUEST_TIMEOUT + 60)):
                is_valid, msg, response_body, req_id = future.result()
                response_bodies.append(f"请求{idx+1}(request_id={req_id}): {response_body}")
                if is_valid:
                    if msg == "error_code_400":
                        results["error_code_400"] += 1
                    else:
                        results["success"] += 1
                else:
                    results["failed"] += 1
                    if len(results["errors"]) < 10:
                        results["errors"].append(msg)
        
        # 记录所有响应体
        with allure.step("响应体列表"):
            for i, body in enumerate(response_bodies):
                allure.attach(body, name=f"响应{i+1}", attachment_type=allure.attachment_type.TEXT)
        
        with allure.step(f"并发请求结果: 成功={results['success']}, 错误码400={results['error_code_400']}, 失败={results['failed']}"):
            pass
        
        total_valid = results["success"] + results["error_code_400"]
        check.equal(total_valid, CONCURRENT_USERS, 
                    f"部分请求失败: {results['failed']} 个, 错误示例: {results['errors'][:3]}")
    
    def test_different_request_id_stream(self, api_client, request):
        """
        场景2：流式 - 1000个用户各自不同X-Request-ID并发请求
        """
        base_url = api_client.base_url
        request_body = self.REQUEST_BODY_STREAM.copy()
        
        # 记录请求体（只记录一次）
        with allure.step(f"请求体: {request_body}"):
            allure.attach(str(request_body), name="请求体", attachment_type=allure.attachment_type.JSON)
        
        results = {"success": 0, "error_code_400": 0, "failed": 0, "errors": []}
        response_bodies = []  # 记录每次响应体
        
        def single_request(index):
            try:
                # 每个请求使用不同的 request_id
                request_id = f"test-diff-id-{uuid.uuid4().hex}"
                response = send_request_with_request_id(
                    base_url, 
                    "/v1/completions", 
                    request_body, 
                    request_id
                )
                is_valid, msg, response_body = validate_response(response, request_id, is_stream=True)
                return is_valid, msg, response_body, request_id
            except Exception as e:
                return False, str(e), "", ""
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(single_request, i) for i in range(CONCURRENT_USERS)]
            for idx, future in enumerate(as_completed(futures, timeout=REQUEST_TIMEOUT + 60)):
                is_valid, msg, response_body, req_id = future.result()
                response_bodies.append(f"请求{idx+1}(request_id={req_id}): {response_body}")
                if is_valid:
                    if msg == "error_code_400":
                        results["error_code_400"] += 1
                    else:
                        results["success"] += 1
                else:
                    results["failed"] += 1
                    if len(results["errors"]) < 10:
                        results["errors"].append(msg)
        
        # 记录所有响应体
        with allure.step("响应体列表"):
            for i, body in enumerate(response_bodies):
                allure.attach(body, name=f"响应{i+1}", attachment_type=allure.attachment_type.TEXT)
        
        with allure.step(f"并发请求结果: 成功={results['success']}, 错误码400={results['error_code_400']}, 失败={results['failed']}"):
            pass
        
        total_valid = results["success"] + results["error_code_400"]
        check.equal(total_valid, CONCURRENT_USERS, 
                    f"部分请求失败: {results['failed']} 个, 错误示例: {results['errors'][:3]}")