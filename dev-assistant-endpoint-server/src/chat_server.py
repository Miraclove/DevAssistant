import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from util import logger
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from marko.ext.gfm import gfm
from sql import MySQLDatabase

# init and server settings
db = MySQLDatabase(host='192.168.103.165', user='uework', password="uework@2022", database='luacoderdb',port=13306)

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# api defination
@app.get("/health")
def read_health():
    """
    Returns a dictionary containing the status of the server.

    Returns:
        dict: A dictionary containing the status of the server.
    """
    return {"status": "healthy"}


@app.post("/api/chat/")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - inputs: the input text to use for the generation.
    - parameters: a dictionary of sampling parameters (See `SamplingParams` for details).
    - stream: whether to stream the results or not.

    Returns a JSON object with the following fields:
    - generated_text: a list of generated text outputs.
    - status: the HTTP status code (200 for success).

    If `stream` is True, the response is a StreamingResponse that yields a JSON object for each generated text output.
    Otherwise, the response is a JSONResponse that contains all generated text outputs.
    """
    # function code here
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    # pre process
    json_request: dict = await request.json()
    logger.info(f'{request.client.host}:{request.client.port} receive json = {json.dumps(json_request)}')

    inputs: str = json_request['inputs']
    parameters: dict = json_request['parameters']
    prompt = inputs
    prompt_template = "<|system|>\n<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>"
    prompt = prompt_template.format(query=prompt)

    # set stream
    stream = False
    if "stream" in parameters.keys():
        stream = parameters.pop("stream", False)
    
    sampling_params = SamplingParams(**parameters,skip_special_tokens=False,stop='<|end|>')
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                gfm(output.text) for output in request_output.outputs
            ]
            ret = {"generated_text": text_outputs,"status": 200}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"generated_text": text_outputs,"status": 200}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=7088)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    model_path = "./models/luachat"
    engine_args = AsyncEngineArgs(model=model_path,gpu_memory_utilization=0.5)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)