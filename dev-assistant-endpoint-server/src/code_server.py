import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from generators import LuaCoder,GeneratorBase
import json
from util import logger, get_parser
from sql import MySQLDatabase


app = FastAPI()
app.add_middleware(
    CORSMiddleware
)

db = MySQLDatabase(host='192.168.103.165', user='uework', password="uework@2022", database='luacoderdb',port=13306)
generator: GeneratorBase
# generator: GeneratorBase = LuaChat()
@app.get("/health")
def read_health():
    """
    Returns a dictionary with the status of the server.

    Returns:
        dict: A dictionary with the status of the server.
    """
    return {"status": "healthy"}

@app.post("/api/generate/")
async def api(request: Request):
    """
    This function handles incoming requests to the server and generates text based on the inputs and parameters provided.

    Args:
        request (Request): The incoming request object.

    Returns:
        dict: A dictionary containing the generated text and the status code.
    """
    # pre process
        # pre process
    json_request: dict = await request.json()
    logger.info(f'{request.client.host}:{request.client.port} receive json = {json.dumps(json_request)}')

    # collect user data, only save lua file
    if 'type' in json_request.keys() and 'filename' in json_request.keys() and 'inputs' in json_request.keys():
        type = json_request['type']
        filename = json_request['filename']
        if type == 'save' and filename.endswith('.lua'):
            user_name = f'{request.client.host}'
            content = json_request['inputs']
            db.insert_or_update(filename, user_name, content)
            logger.info(f'{request.client.host}:{request.client.port} save file = {filename}')
            return {"generated_text": "","status": 200}
    

    inputs: str = json_request['inputs']
    parameters: dict = json_request['parameters']
    generated_text: str = generator.generate(inputs, parameters)
    logger.info(f'{request.client.host}:{request.client.port} generated_text = {json.dumps(generated_text)}')
    return {
        "generated_text": generated_text,
        "status": 200
    }

def main():
    """
    Runs the endpoint server using the specified host and port.

    :return: None
    """
    global generator
    generator = LuaCoder()
    args = get_parser().parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()