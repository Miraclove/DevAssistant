import argparse
import json

import gradio as gr
import requests


import requests
import json

def http_bot(prompt):
    """
    Sends a prompt to a remote model via HTTP and yields the model's response.

    Args:
        prompt (str): The prompt to send to the model.

    Yields:
        str: The model's response to the prompt.
    """
    headers = {"User-Agent": "vLLM Client"}
    pload = {
        "inputs": prompt,
        "parameters": {
            "stream": True,
            "max_tokens": 128,
        }
    }
    response = requests.post(args.model_url,
                             headers=headers,
                             json=pload,
                             stream=True)

    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"][0]
            yield output


def build_demo():
    """
    Builds a Gradio demo for the vLLM text completion model.

    Returns:
    gradio.Interface: A Gradio interface object for the vLLM text completion demo.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# vLLM text completion demo\n")
        inputbox = gr.Textbox(label="Input",
                              placeholder="Enter text and press ENTER")
        outputbox = gr.Textbox(label="Output",
                               placeholder="Generated result from the model")
        inputbox.submit(http_bot, [inputbox], [outputbox])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model-url",
                        type=str,
                        default="http://localhost:8000/generate")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue(concurrency_count=100).launch(server_name=args.host,
                                             server_port=args.port,
                                             share=True)