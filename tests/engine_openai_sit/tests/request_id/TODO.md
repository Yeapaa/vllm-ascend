·v1/chat/completions、v1/completions·2个接口支持通过request header字段"X-Request-ID"传递request_id，用于trace请求链路。
现在针对每个接口的请求体stream的true/false 流式/非流式开关，分别需要补充2个场景，用于测试request_id是否正确传递。
### 第1个场景：✅
多进程多线程多协程都行，模拟1000个用户以相同X-Request-ID并发请求，响应可以是error code 400，如果不是error code 400，则正常响应，添加校验finish_reason为stop或length，校验响应体字段id值以请求的X-Request-ID值为后缀。


### 第2个场景：✅
多进程多线程多协程都行，模拟1000个用户各自不同X-Request-ID并发请求，正常响应，添加校验finish_reason为stop或length，校验响应体字段id值以请求的X-Request-ID值为后缀。


### 第3个场景？：⚠️效果层面，待定。
多进程多线程多协程都行，模拟1000个用户各自不同X-Request-ID并发请求，正常响应，多轮对话，检测响应对应请求。