"""
NanoLM Web 推理服务

用法:
    python -m toolkit web
"""

import json
import os
import sys
import traceback

import torch
from bottle import Bottle, request, response, run

from model import LlmModel
from nanolm.device import get_optimal_device
from nanolm.utils import get_model_config, init_env
from train import TrainerTools, streaming_generate


def main():
    init_env()
    device = get_optimal_device()

    os.makedirs('./bin/', exist_ok=True)
    os.makedirs('./cache/', exist_ok=True)
    model_name = 'ppo_policy.bin'

    if not os.path.exists(f'./bin/{model_name}'):
        raise FileNotFoundError(f"Model file {model_name} not found in ./bin/")

    model = LlmModel(get_model_config(long_context=True)).to(device=device)
    model.load_state_dict(torch.load(f'./bin/{model_name}', weights_only=True))
    model.eval()

    system_tokens = TrainerTools().tokenizer.encode('<system> </s>')
    max_new_tokens = 512
    max_user_tokens = 2048 - max_new_tokens

    app = Bottle()

    static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
    with open(os.path.join(static_dir, 'index.html')) as f:
        html = f.read()

    def fmt_msg(event, data):
        data = data.replace('\n', '<br />')
        return f"{json.dumps({'event': event, 'data': data})}\n\n"

    @app.get('/')
    def index():
        visitor_count = 0
        generate_count = 0

        if os.path.exists('./cache/visitor.txt'):
            with open('./cache/visitor.txt') as f:
                visitor_count = int(f.readline())

        if os.path.exists('./cache/generator.txt'):
            with open('./cache/generator.txt') as f:
                generate_count = int(f.readline())

        with open('./cache/visitor.txt', 'w') as f:
            f.write(f'{visitor_count + 1}')

        return html.replace('{{__VISITOR_COUNT__}}', f"{visitor_count}").replace('{{__GENERATE_COUNT__}}', f"{generate_count}")

    @app.hook('after_request')
    def enable_cors():
        """Add CORS headers to all responses to allow cross-origin requests."""
        response.set_header('Access-Control-Allow-Origin', '*')
        response.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        response.set_header('Access-Control-Allow-Headers', 'Content-Type')

    @app.route('/api/chat', method=['OPTIONS'])
    def options_handler():
        """Handle CORS pre-flight requests."""
        return

    @app.route('/api/chat', method=['POST'])
    def sse_chat():
        """
        Handles chat requests and returns a streaming response via SSE.
        """
        response.content_type = 'text/event-stream'
        response.set_header('Cache-Control', 'no-cache')
        response.set_header('Connection', 'keep-alive')

        try:
            generate_count = 0
            if os.path.exists('./cache/generator.txt'):
                with open('./cache/generator.txt') as f:
                    generate_count = int(f.readline())

            with open('./cache/generator.txt', 'w') as f:
                f.write(f'{generate_count + 1}')

            payload = request.json
            chat_history: list = payload.get('history')
            temperature = payload.get('temperature')
            top_p = payload.get('top_p')

            if not chat_history:
                yield fmt_msg('error', 'Chat history cannot be empty')
                return

        except (json.JSONDecodeError, AttributeError):
            yield fmt_msg('error', 'Invalid JSON payload')
            return

        try:
            chat_history.reverse()
            chat_tokens = []

            for chat in chat_history:
                role = '<user>' if chat['role'] == 'user' else '<assistant>'
                chat_item_tokens = TrainerTools().tokenizer.encode(f"{role}{chat['content']}</s>")
                if len(system_tokens) + len(chat_tokens) + len(chat_item_tokens) >= max_user_tokens:
                    break
                chat_tokens.append(chat_item_tokens)

            chat_tokens.reverse()
            chat_tokens = [item for sublist in chat_tokens for item in sublist]
            chat_tokens.append(TrainerTools().tokenizer.assistant)
            chat_tokens = system_tokens + chat_tokens

            print(TrainerTools().tokenizer.decode(chat_tokens))

            generator = streaming_generate(
                model=model,
                prompt=torch.tensor(chat_tokens),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                k=None,
                p=top_p,
                device=device,
                return_token=True
            )

            all_response_tokens = []
            for chunk in generator:
                if chunk == TrainerTools().tokenizer.end:
                    break
                all_response_tokens.append(chunk)
                yield fmt_msg('answer_chunk', TrainerTools().tokenizer.decode(torch.tensor(all_response_tokens)))

        except Exception as e:
            traceback.print_exc()
            print(f"Error during model generation: {e}")
            yield fmt_msg('error', f'Internal server error: {e}')

    use_waitress = os.environ.get('USE_WAITRESS', '') == '1'

    if use_waitress:
        try:
            from waitress import serve
        except ImportError:
            print("Waitress not installed. Run: pip3 install waitress")
            sys.exit(1)

        print("Server ready! Open http://localhost:8080 in your browser.")
        serve(app, host='0.0.0.0', port=8080)
    else:
        print("Server ready! Open http://localhost:8080 in your browser.")
        run(app, host='0.0.0.0', port=8080, server='paste')


if __name__ == '__main__':
    main()
