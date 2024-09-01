# OpenAI API python demo

api doc: https://platform.openai.com/docs/api-reference

## chat.completions

```
export OPENAI_LOG=debug
```

```
import os
from openai import OpenAI

api_key = "xxx.xxx"
server_url = "https://open.bigmodel.cn/api/paas/v4"

client = OpenAI(
    api_key=api_key,
    base_url=server_url
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="glm-4-air",
)

# [2024-06-15 03:44:12 - openai._base_client:446 - DEBUG] Request options: {'method': 'post', 'url': '/chat/completions', 'files': None, 'json_data': {'messages': [{'role': 'user', 'content': 'Say this is a test'}], 'model': 'glm-4-air'}}

# ChatCompletion(id='8741753714584919126', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Understood. This is a test response. How can I assist you further?', role='assistant', function_call=None, tool_calls=None))], created=1718421897, model='glm-4-air', object=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=18, prompt_tokens=10, total_tokens=28), request_id='8741753714584919126')

chat_completion.to_dict()
#{'id': '8741753714584919126', 'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'content': 'Understood. This is a test response. How can I assist you further?', 'role': 'assistant'}}], 'created': 1718421897, 'model': 'glm-4-air', 'usage': {'completion_tokens': 18, 'prompt_tokens': 10, 'total_tokens': 28}, 'request_id': '8741753714584919126'}

{'choices': [{'finish_reason': 'stop',
              'index': 0,
              'message': {'content': 'Understood. This is a test response. How '
                                     'can I assist you further?',
                          'role': 'assistant'}}],
 'created': 1718421897,
 'id': '8741753714584919126',
 'model': 'glm-4-air',
 'request_id': '8741753714584919126',
 'usage': {'completion_tokens': 18, 'prompt_tokens': 10, 'total_tokens': 28}}
```

## tools

```
import os
import openai
from openai import OpenAI

get_weather =  {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }

api_key = "xxx.xxx"
server_url = "https://open.bigmodel.cn/api/paas/v4"

client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
    base_url=server_url
)

tools = [
    {
        "type": "function",
        "function":get_weather
    }
]

messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston?"
    }
]

response = client.chat.completions.create(
    messages=messages,
    model="glm-4-air",
    tools=tools,
)

# [2024-06-15 03:48:16 - openai._base_client:446 - DEBUG] Request options: {'method': 'post', 'url': '/chat/completions', 'files': None, 'json_data': {'messages': [{'role': 'user', 'content': "What's the weather like in Boston?"}], 'model': 'glm-4-air', 'tools': [{'type': 'function', 'function': {'name': 'get_current_weather', 'description': 'Get the current weather in a given location', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'}, 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['location']}}}]}}

# ChatCompletion(id='8741754470499318134', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_8741754470499318134', function=Function(arguments='{"location": "Boston", "unit": "celsius"}', name='get_current_weather'), type='function', index=0)]))], created=1718423297, model='glm-4-air', object=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=18, prompt_tokens=206, total_tokens=224), request_id='8741754470499318134')

{
  "id": "8741754470499318134",
  "choices": [
    {
      "finish_reason": "tool_calls",
      "index": 0,
      "message": {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_8741754470499318134",
            "function": {
              "arguments": "{\"location\": \"Boston\", \"unit\": \"celsius\"}",
              "name": "get_current_weather"
            },
            "type": "function",
            "index": 0
          }
        ]
      }
    }
  ],
  "created": 1718423297,
  "model": "glm-4-air",
  "usage": {
    "completion_tokens": 18,
    "prompt_tokens": 206,
    "total_tokens": 224
  },
  "request_id": "8741754470499318134"
}

```

## Restful API demo

```
import json
import requests
import base64
from io import BytesIO
from PIL import Image

headers = {
    'Authorization': 'Bearer sk-xxx'
}

model_name = 'xxx'
api_url = 'http://xxx/v1/chat/completions'

# 图片转base64
def img_to_base64(img_path):
    with open(img_path,"rb") as f:
        base64_str = base64.b64encode(f.read()).decode("utf-8")
    return base64_str

# 构造对话消息
def construct_message(prompt, img_file):
    text =  {"type": "text", "text": prompt}
    if img_file:
        b64 = img_to_base64(img_file)
        image = {"type": "image_url",
                 "image_url": { "url": f"data:image/jpeg;base64,{b64}"}}
        messages = [{"role": "user", "content": [text, image]}]
    else:
        messages = [{"role": "user", "content": [text]}]
    return messages


def get_response(prompt, img_file=None, stream=True):
    messages = construct_message(prompt, img_file)
    datas = {"model": model_name, "messages": messages,  "temperature": 0.7, 
             "stream": stream,   "max_tokens":2048}
    response = requests.post(url=api_url, headers=headers, data=json.dumps(datas), 
               stream=stream)
    if response.status_code != 200:
        raise ValueError("Failed to generate response: " + response.text)
    if stream:
        result = []
        for chunk in response.iter_lines():
            if not chunk:
                continue
            json_str = chunk.decode('utf-8')
            if json_str[6:] == "[DONE]":
                break
            chunk = json.loads(json_str[6:])
            if "choices" in chunk and len(chunk["choices"]) > 0:
                choice = chunk["choices"][0]
                if "delta" in choice:
                    print(choice["delta"]["content"], end ="", flush=True)
                    result.append(choice["delta"]["content"])
        print("")
        return "".join(result)
    else:
        response = response.json()
        res = response['choices'][0]['message']['content']
        print(res)
        return res

def chat(question, img_file=None):
    response = get_response(question, img_file)
    return response

if __name__ == '__main__':
    image_file = 'xxx'
    question = "请详细描述图片"

    chat(question, image_file)
```