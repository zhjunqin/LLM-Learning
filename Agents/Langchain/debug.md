# LangChain Debug

LangChain Debug 的方式

## debug = True

```
import langchain
langchain.debug = True
```

在执行中会打印类似如下日志：

```
[chain/start] [chain:LangGraph] Entering Chain run with input:
[inputs]
[chain/start] [chain:LangGraph > chain:__start__] Entering Chain run with input:
[inputs]
[chain/end] [chain:LangGraph > chain:__start__] [1ms] Exiting Chain run with output:
[outputs]
[chain/start] [chain:LangGraph > chain:llm] Entering Chain run with input:
[inputs]
[llm/start] [chain:LangGraph > chain:llm > llm:ChatOpenAI] Entering LLM run with input:
{
  "prompts": [
    "System: You are a smart research assistant. Use the search engine to look up information. You are allowed to make multiple calls (either together or in sequence). Only look up information when you are sure of what you want. If you need to look up some information before asking a follow up question, you are allowed to do that!\n\nHuman: What is the weather in sf?"
  ]
}
[llm/end] [chain:LangGraph > chain:llm > llm:ChatOpenAI] [702ms] Exiting LLM run with output:
{
  "generations": [
    [
      {
        "text": "",
        "generation_info": {
          "finish_reason": "tool_calls",
          "logprobs": null
        },
        "type": "ChatGeneration",
        "message": {
          "lc": 1,
          "type": "constructor",
          "id": [
            "langchain",
            "schema",
            "messages",
            "AIMessage"
          ],
          "kwargs": {
            "content": "",
            "additional_kwargs": {
              "tool_calls": [
                {
                  "id": "call_pMDzDi8MCGNZc4QVTtUIkq2P",
                  "function": {
                    "arguments": "{\"query\":\"current weather in San Francisco\"}",
                    "name": "tavily_search_results_json"
                  },
                  "type": "function"
                }
              ]
            },
            "response_metadata": {
              "token_usage": {
                "completion_tokens": 22,
                "prompt_tokens": 153,
                "total_tokens": 175
              },
              "model_name": "gpt-3.5-turbo",
              "system_fingerprint": null,
              "finish_reason": "tool_calls",
              "logprobs": null
            },
            "type": "ai",
            "id": "run-ff5693a2-46b2-405d-ac10-adb422d77738-0",
            "tool_calls": [
              {
                "name": "tavily_search_results_json",
                "args": {
                  "query": "current weather in San Francisco"
                },
                "id": "call_pMDzDi8MCGNZc4QVTtUIkq2P"
              }
            ],
            "invalid_tool_calls": []
          }
        }
      }
    ]
  ],
  "llm_output": {
    "token_usage": {
      "completion_tokens": 22,
      "prompt_tokens": 153,
      "total_tokens": 175
    },
    "model_name": "gpt-3.5-turbo",
    "system_fingerprint": null
  },
  "run": null
}
[chain/start] [chain:LangGraph > chain:llm > chain:ChannelWrite<llm,messages>] Entering Chain run with input:
[inputs]
[chain/end] [chain:LangGraph > chain:llm > chain:ChannelWrite<llm,messages>] [0ms] Exiting Chain run with output:
[outputs]
[chain/start] [chain:LangGraph > chain:llm > chain:exists_action] Entering Chain run with input:
[inputs]
[chain/end] [chain:LangGraph > chain:llm > chain:exists_action] [0ms] Exiting Chain run with output:
{
  "output": true
}
[chain/start] [chain:LangGraph > chain:llm > chain:ChannelWrite<branch:llm:exists_action:action>] Entering Chain run with input:
[inputs]
[chain/end] [chain:LangGraph > chain:llm > chain:ChannelWrite<branch:llm:exists_action:action>] [0ms] Exiting Chain run with output:
[outputs]
[chain/end] [chain:LangGraph > chain:llm] [707ms] Exiting Chain run with output:
[outputs]
[chain/start] [chain:LangGraph > chain:action] Entering Chain run with input:
[inputs]
Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_pMDzDi8MCGNZc4QVTtUIkq2P'}
[tool/start] [chain:LangGraph > chain:action > tool:tavily_search_results_json] Entering Tool run with input:
"{'query': 'current weather in San Francisco'}"
Error in ConsoleCallbackHandler.on_tool_end callback: AttributeError("'list' object has no attribute 'strip'")
Back to the model!
[chain/start] [chain:LangGraph > chain:action > chain:ChannelWrite<action,messages>] Entering Chain run with input:
[inputs]
[chain/end] [chain:LangGraph > chain:action > chain:ChannelWrite<action,messages>] [0ms] Exiting Chain run with output:
[outputs]
[chain/end] [chain:LangGraph > chain:action] [1.94s] Exiting Chain run with output:
[outputs]
[chain/start] [chain:LangGraph > chain:llm] Entering Chain run with input:
[inputs]
[llm/start] [chain:LangGraph > chain:llm > llm:ChatOpenAI] Entering LLM run with input:
{
  "prompts": [
    "System: You are a smart research assistant. Use the search engine to look up information. You are allowed to make multiple calls (either together or in sequence). Only look up information when you are sure of what you want. If you need to look up some information before asking a follow up question, you are allowed to do that!\n\nHuman: What is the weather in sf?\nAI: \nTool: [{'url': 'https://www.weatherapi.com/', 'content': \"{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.78, 'lon': -122.42, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1718010268, 'localtime': '2024-06-10 2:04'}, 'current': {'last_updated_epoch': 1718010000, 'last_updated': '2024-06-10 02:00', 'temp_c': 12.8, 'temp_f': 55.0, 'is_day': 0, 'condition': {'text': 'Overcast', 'icon': '//cdn.weatherapi.com/weather/64x64/night/122.png', 'code': 1009}, 'wind_mph': 13.6, 'wind_kph': 22.0, 'wind_degree': 250, 'wind_dir': 'WSW', 'pressure_mb': 1015.0, 'pressure_in': 29.98, 'precip_mm': 0.01, 'precip_in': 0.0, 'humidity': 93, 'cloud': 100, 'feelslike_c': 11.4, 'feelslike_f': 52.4, 'windchill_c': 10.5, 'windchill_f': 50.9, 'heatindex_c': 12.0, 'heatindex_f': 53.7, 'dewpoint_c': 10.1, 'dewpoint_f': 50.2, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 1.0, 'gust_mph': 18.1, 'gust_kph': 29.2}}\"}, {'url': 'https://forecast.weather.gov/zipcity.php?inputstring=San+Francisco,CA', 'content': 'San Francisco CA 37.77°N 122.41°W (Elev. 131 ft) Last Update: 6:00 pm PDT Jun 7, 2024. Forecast Valid: 7pm PDT Jun 7, 2024-6pm PDT Jun 14, 2024 . Forecast Discussion . Additional Resources. Radar & Satellite Image. Hourly Weather Forecast. ... Severe Weather ; Current Outlook Maps ; Drought ; Fire Weather ; Fronts/Precipitation Maps ; Current ...'}, {'url': 'https://www.wunderground.com/hourly/us/ca/san-francisco/94111/date/2024-6-10', 'content': 'San Francisco Weather Forecasts. Weather Underground provides local & long-range weather forecasts, weatherreports, maps & tropical weather conditions for the San Francisco area.'}, {'url': 'https://weather.com/weather/tenday/l/San Francisco CA USCA0987:1:US', 'content': \"Comfy & Cozy\\nThat's Not What Was Expected\\nOutside\\n'No-Name Storms' In Florida\\nGifts From On High\\nWhat To Do For Wheezing\\nSurviving The Season\\nStay Safe\\nAir Quality Index\\nAir quality is considered satisfactory, and air pollution poses little or no risk.\\n Health & Activities\\nSeasonal Allergies and Pollen Count Forecast\\nNo pollen detected in your area\\nCold & Flu Forecast\\nFlu risk is low in your area\\nWe recognize our responsibility to use data and technology for good. recents\\nSpecialty Forecasts\\n10 Day Weather-San Francisco, CA\\nToday\\nMon 18 | Day\\nConsiderable cloudiness. Tue 19\\nTue 19 | Day\\nLight rain early...then remaining cloudy with showers in the afternoon. Wed 27\\nWed 27 | Day\\nOvercast with rain showers at times.\"}]"
  ]
}
[llm/end] [chain:LangGraph > chain:llm > llm:ChatOpenAI] [1.15s] Exiting LLM run with output:
{
  "generations": [
    [
      {
        "text": "The current weather in San Francisco is overcast with a temperature of 55.0°F (12.8°C). The wind speed is 13.6 mph (22.0 kph) coming from the WSW direction. The humidity is at 93%, and the visibility is 9.0 miles.",
        "generation_info": {
          "finish_reason": "stop",
          "logprobs": null
        },
        "type": "ChatGeneration",
        "message": {
          "lc": 1,
          "type": "constructor",
          "id": [
            "langchain",
            "schema",
            "messages",
            "AIMessage"
          ],
          "kwargs": {
            "content": "The current weather in San Francisco is overcast with a temperature of 55.0°F (12.8°C). The wind speed is 13.6 mph (22.0 kph) coming from the WSW direction. The humidity is at 93%, and the visibility is 9.0 miles.",
            "response_metadata": {
              "token_usage": {
                "completion_tokens": 65,
                "prompt_tokens": 1017,
                "total_tokens": 1082
              },
              "model_name": "gpt-3.5-turbo",
              "system_fingerprint": null,
              "finish_reason": "stop",
              "logprobs": null
            },
            "type": "ai",
            "id": "run-831d7770-8146-49d9-8655-92c2ca285acd-0",
            "tool_calls": [],
            "invalid_tool_calls": []
          }
        }
      }
    ]
  ],
  "llm_output": {
    "token_usage": {
      "completion_tokens": 65,
      "prompt_tokens": 1017,
      "total_tokens": 1082
    },
    "model_name": "gpt-3.5-turbo",
    "system_fingerprint": null
  },
  "run": null
}
[chain/start] [chain:LangGraph > chain:llm > chain:ChannelWrite<llm,messages>] Entering Chain run with input:
[inputs]
[chain/end] [chain:LangGraph > chain:llm > chain:ChannelWrite<llm,messages>] [0ms] Exiting Chain run with output:
[outputs]
[chain/start] [chain:LangGraph > chain:llm > chain:exists_action] Entering Chain run with input:
[inputs]
[chain/end] [chain:LangGraph > chain:llm > chain:exists_action] [0ms] Exiting Chain run with output:
{
  "output": false
}
[chain/end] [chain:LangGraph > chain:llm] [1.16s] Exiting Chain run with output:
[outputs]
[chain/end] [chain:LangGraph] [3.82s] Exiting Chain run with output:
[outputs]
```

## Wrapper

使用 Wrapper

```
llm = ChatOpenAI(model="gpt-3.5-turbo-0613")

# class that wraps another class and logs all function calls being executed 
class Wrapper:
    def __init__(self, wrapped_class):
        self.wrapped_class = wrapped_class

    def __getattr__(self, attr):
        original_func = getattr(self.wrapped_class, attr)

        def wrapper(*args, **kwargs):
            print(f"Calling function: {attr}")
            print(f"Arguments: {args}, {kwargs}")
            result = original_func(*args, **kwargs)
            print(f"Response: {result}")
            return result

        return wrapper

# overwrite the private `client` attribute inside of the LLM that contains the API client with our wrapped class
llm.client = Wrapper(llm.client)
```

可以打印原始的消息：

```
Calling function: create
Arguments: (), {'messages': [{'content': 'You are a smart research assistant. Use the search engine to look up information. You are allowed to make multiple calls (either together or in sequence). Only look up information when you are sure of what you want. If you need to look up some information before asking a follow up question, you are allowed to do that!\n', 'role': 'system'}, {'content': 'What is the weather in sf?', 'role': 'user'}], 'model': 'GLM-4-0520', 'stream': False, 'n': 1, 'temperature': 0.1, 'tools': [{'type': 'function', 'function': {'name': 'tavily_search_results_json', 'description': 'A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.', 'parameters': {'type': 'object', 'properties': {'query': {'description': 'search query to look up', 'type': 'string'}}, 'required': ['query']}}}]}
Response: ChatCompletion(id='8730209598407273114', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_8730209598407273114', function=Function(arguments='{"query": "weather in San Francisco"}', name='tavily_search_results_json'), type='function', index=0)]))], created=1718011404, model='GLM-4-0520', object=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=17, prompt_tokens=215, total_tokens=232), request_id='8730209598407273114')
Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_8730209598407273114'}
Back to the model!
```

## Http Verbose Logging


```
import http
import logging
import requests


http.client.HTTPConnection.debuglevel = 1

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

requests.get('http://secariolabs.com')
```


## 参考文献
- https://github.com/langchain-ai/langchain/issues/6628
- https://github.com/langchain-ai/langchain/discussions/6511
- https://secariolabs.com/logging-raw-http-requests-in-python/