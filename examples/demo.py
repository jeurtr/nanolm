"""
NanoLM 快速体验

运行: python examples/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanolm import chat, generate, load_model


def main():
    print("=" * 50)
    print("NanoLM 快速体验")
    print("=" * 50)

    # 1. 加载模型
    print("\n[1/3] 加载模型 ...")
    model = load_model("./bin/ppo_policy.bin")
    print("模型加载完成")

    # 2. 单轮生成
    print("\n[2/3] 单轮生成 ...")
    prompts = [
        "用一句话介绍人工智能",
        "写一首关于春天的五言绝句",
        "什么是机器学习？请用通俗的语言解释",
    ]
    for p in prompts:
        print(f"\n  Q: {p}")
        print(f"  A: {generate(model, p, max_tokens=256)}")

    # 3. 多轮对话
    print("\n[3/3] 多轮对话 ...")
    history = [
        {"role": "user", "content": "你好，我叫小明"},
        {"role": "assistant", "content": "你好小明！有什么可以帮助你的吗？"},
        {"role": "user", "content": "我叫什么名字？"},
    ]
    response = chat(model, history, max_tokens=128)
    print(f"  对话历史: {[h['content'][:20] for h in history]}")
    print(f"  助手回复: {response}")

    print("\n✓ 体验完成")


if __name__ == '__main__':
    main()
