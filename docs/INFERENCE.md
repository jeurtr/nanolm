# NanoLM 推理服务

## 启动

```bash
python3 app.py
```

模型加载完成后打印:
```
Server ready! Open http://localhost:8080 in your browser.
```

首次运行会自动创建 `cache/` 目录（存放访问计数器）。

---

## 架构

```
Browser (static/index.html)
    │  SSE (Server-Sent Events)
    ▼
Bottle + PasteServer (port 8080)
    │
    ├── GET  /          → 返回聊天页面 HTML
    ├── POST /api/chat  → SSE 流式生成回复
    └── OPTIONS /api/chat → CORS pre-flight
    │
    ▼
LlmModel (80M params, ppo_policy.bin)
    │  MPS / CUDA / CPU
    ▼
streaming_generate() → 逐 token yield
```

---

## 模型加载

```python
device = "cuda" if GPU else "mps" if Apple Silicon else "cpu"
model = LlmModel(get_model_config(long_context=True)).to(device)
model.load_state_dict(torch.load('./bin/ppo_policy.bin'))
model.eval()
```

- 模型文件: `./bin/ppo_policy.bin` (327 MB)
- 上下文窗口: 2048 tokens
- 最大生成: 512 tokens
- 用户历史上限: 2048 - 512 = 1536 tokens

---

## API 接口

### POST /api/chat

**请求格式**:
```json
{
  "history": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
    {"role": "user", "content": "介绍一下太阳系"}
  ],
  "temperature": 1.0,
  "top_p": 0.95
}
```

**响应格式** (SSE 流):
```
{"event": "answer_chunk", "data": "太"}\n\n
{"event": "answer_chunk", "data": "太阳"}\n\n
{"event": "answer_chunk", "data": "太阳系"}\n\n
...
```

**错误响应**:
```
{"event": "error", "data": "Chat history cannot be empty"}\n\n
{"event": "error", "data": "Invalid JSON payload"}\n\n
{"event": "error", "data": "Internal server error: ..."}\n\n
```

---

## 对话处理流程

```
1. 接收 Chat History（JSON）
     ↓
2. 反转历史 → 从最新到最旧构建 token 序列
     ↓
3. 限制总长度 ≤ 1536 tokens
     ↓
4. 拼接格式:
   <system> </s><user>msg1</s><assistant>reply1</s><user>msg2</s><assistant>
     ↓
5. streaming_generate() → 逐 token 生成
     ↓
6. 累积解码 → SSE answer_chunk 事件
     ↓
7. 遇到 </s> → 停止
```

---

## 生成参数

### Temperature

控制输出随机性:
- 低值 (0.1-0.3): 确定性输出，适合事实性问答
- 中值 (0.7-1.0): 平衡创造性和准确性
- 高值 (1.2-2.0): 更多随机性，适合创意写作

**默认**: 1.0

### Top-p (Nucleus Sampling)

只从累积概率达到 p 的 token 中采样:
- 低值 (0.1-0.3): 保守，只考虑高概率 token
- 中值 (0.7-0.9): 平衡
- 高值 (0.95-1.0): 考虑更多可能性

**默认**: 0.95

### Top-k

只从概率最高的 k 个 token 中采样。

**默认**: None（不使用）

---

## 生成实现

### streaming_generate()

```python
def streaming_generate(model, prompt, max_new_tokens, temperature, k, p, device, return_token):
    """
    逐 token 生成，每次 yield 新 token。

    内部流程:
    1. tokenize prompt
    2. for _ in range(max_new_tokens):
         a. model.forward() + KV Cache
         b. 应用 temperature / top-k / top-p warpers
         c. multinomial 采样 或 argmax
         d. yield next_token
         e. 更新 KV Cache
         f. 如果 token == </s> → break
    """
```

### Token Warpers (按顺序应用)

1. **suppress**: 将指定 token 的 logits 设为 -inf
2. **temperature**: `logits /= temperature`
3. **top-k**: 只保留概率最高的 k 个 token
4. **top-p**: 核采样，保留累积概率达到 p 的 token

### batch_generate() (用于 PPO rollout)

- 预分配 `(batch, max_new_tokens)` 缓冲区
- 追踪 per-sample `done` mask
- 支持返回 per-step logits（用于 RL advantage 计算）
- 使用 KV Cache + position ID 追踪

---

## 前端

`static/index.html` — 单页聊天应用：
- Tailwind CSS 样式
- Markdown 渲染 (marked.js)
- 代码高亮 (highlight.js)
- 数学公式渲染 (KaTeX)
- SSE 流式接收
- Temperature / Top-p 滑块调节
- 访客计数显示

### 服务器选择

| 环境变量 | 服务器 | 说明 |
|---|---|---|
| 默认 | PasteServer | Bottle 内置，支持 SSE 流式 |
| `USE_WAITRESS=1` | Waitress | 需安装 `pip3 install waitress` |

---

## 推理性能

| 平台 | 设备 | 速度 |
|---|---|---|
| NVIDIA GPU | CUDA | 最快 |
| Apple Silicon | MPS | 中等 |
| x86/ARM | CPU | 较慢 |

模型总参数量 82M，单次推理显存占用约 400MB (BF16)。
