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
