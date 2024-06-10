# Persistence and Streaming

消息持久化和流式输出

## 消息持久化

```
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults


tool = TavilySearchResults(max_results=2)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:") # 增加 memory


class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer) # 使用 checkpointer
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}


prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatOpenAI(model="gpt-4o")
abot = Agent(model, [tool], system=prompt, checkpointer=memory) # 传入 memory

messages = [HumanMessage(content="What is the weather in sf?")]

thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread): # 使用 thread
    for v in event.values():
        print(v['messages'])
```

输出内容
```
[AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_FIYdoCyQTkTDKCqi62wGqo4h', 'function': {'arguments': '{"query":"current weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 151, 'total_tokens': 173}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_319be4768e', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-c38a6d51-87a6-4319-a545-64dcdd4bbf9d-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_FIYdoCyQTkTDKCqi62wGqo4h'}])]
Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_FIYdoCyQTkTDKCqi62wGqo4h'}
Back to the model!
[ToolMessage(content='[{\'url\': \'https://www.weatherapi.com/\', \'content\': "{\'location\': {\'name\': \'San Francisco\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 37.78, \'lon\': -122.42, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1718012036, \'localtime\': \'2024-06-10 2:33\'}, \'current\': {\'last_updated_epoch\': 1718011800, \'last_updated\': \'2024-06-10 02:30\', \'temp_c\': 12.8, \'temp_f\': 55.0, \'is_day\': 0, \'condition\': {\'text\': \'Overcast\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/night/122.png\', \'code\': 1009}, \'wind_mph\': 16.1, \'wind_kph\': 25.9, \'wind_degree\': 250, \'wind_dir\': \'WSW\', \'pressure_mb\': 1015.0, \'pressure_in\': 29.97, \'precip_mm\': 0.01, \'precip_in\': 0.0, \'humidity\': 93, \'cloud\': 100, \'feelslike_c\': 11.4, \'feelslike_f\': 52.4, \'windchill_c\': 10.5, \'windchill_f\': 50.9, \'heatindex_c\': 12.0, \'heatindex_f\': 53.7, \'dewpoint_c\': 10.1, \'dewpoint_f\': 50.2, \'vis_km\': 16.0, \'vis_miles\': 9.0, \'uv\': 1.0, \'gust_mph\': 20.6, \'gust_kph\': 33.1}}"}, {\'url\': \'https://forecast.weather.gov/MapClick.php?x=165&y=156&site=mtr&zmx=&map_x=165&map_y=156\', \'content\': \'2 Miles S San Francisco CA 37.77°N 122.44°W (Elev. 253 ft) Last Update: 1:27 pm PDT Jun 6, 2024. Forecast Valid: 8pm PDT Jun 6, 2024-6pm PDT Jun 13, 2024 . Forecast Discussion . Additional Resources. Radar & Satellite Image. Hourly Weather Forecast ... Severe Weather ; Current Outlook Maps ; Drought ; Fire Weather ; Fronts/Precipitation Maps ...\'}]', name='tavily_search_results_json', tool_call_id='call_FIYdoCyQTkTDKCqi62wGqo4h')]
[AIMessage(content='The current weather in San Francisco is overcast with a temperature of 12.8°C (55.0°F). The wind is blowing from the west-southwest (WSW) at 16.1 mph (25.9 kph). The humidity is at 93%, and the visibility is 16 kilometers (9 miles).', response_metadata={'token_usage': {'completion_tokens': 70, 'prompt_tokens': 739, 'total_tokens': 809}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_319be4768e', 'finish_reason': 'stop', 'logprobs': None}, id='run-6b8d703a-e575-4939-921a-3d708adaf04e-0')]
```

追问新的问题：

```
messages = [HumanMessage(content="What about in la?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
```
输出内容

```
{'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_uz5oaz0EfrD96wKmkSjOvXfS', 'function': {'arguments': '{"query":"current weather in Los Angeles"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 821, 'total_tokens': 843}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_319be4768e', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-166da5aa-19ab-4838-96d8-b73a086f08ea-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Los Angeles'}, 'id': 'call_uz5oaz0EfrD96wKmkSjOvXfS'}])]}
Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Los Angeles'}, 'id': 'call_uz5oaz0EfrD96wKmkSjOvXfS'}
Back to the model!
{'messages': [ToolMessage(content='[{\'url\': \'https://www.weatherapi.com/\', \'content\': "{\'location\': {\'name\': \'Los Angeles\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 34.05, \'lon\': -118.24, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1718013003, \'localtime\': \'2024-06-10 2:50\'}, \'current\': {\'last_updated_epoch\': 1718012700, \'last_updated\': \'2024-06-10 02:45\', \'temp_c\': 16.7, \'temp_f\': 62.1, \'is_day\': 0, \'condition\': {\'text\': \'Overcast\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/night/122.png\', \'code\': 1009}, \'wind_mph\': 2.2, \'wind_kph\': 3.6, \'wind_degree\': 10, \'wind_dir\': \'N\', \'pressure_mb\': 1014.0, \'pressure_in\': 29.94, \'precip_mm\': 0.0, \'precip_in\': 0.0, \'humidity\': 84, \'cloud\': 100, \'feelslike_c\': 16.7, \'feelslike_f\': 62.1, \'windchill_c\': 17.3, \'windchill_f\': 63.1, \'heatindex_c\': 17.3, \'heatindex_f\': 63.1, \'dewpoint_c\': 12.5, \'dewpoint_f\': 54.5, \'vis_km\': 16.0, \'vis_miles\': 9.0, \'uv\': 1.0, \'gust_mph\': 3.0, \'gust_kph\': 4.9}}"}, {\'url\': \'https://forecast.weather.gov/MapClick.php?lat=33.9425&lon=-118.409\', \'content\': \'Downtown Los Angeles Weather Station has Moved; ... Current conditions at Los Angeles, Los Angeles International Airport (KLAX) Lat: 33.93806°NLon: 118.38889°WElev: 125.0ft. Overcast. 66°F. 19°C. ... 12pm PDT Jun 6, 2024-6pm PDT Jun 12, 2024 . Forecast Discussion . Additional Resources.\'}]', name='tavily_search_results_json', tool_call_id='call_uz5oaz0EfrD96wKmkSjOvXfS')]}
{'messages': [AIMessage(content='The current weather in Los Angeles is overcast with a temperature of 16.7°C (62.1°F). The wind is coming from the north (N) at 2.2 mph (3.6 kph). The humidity is at 84%, and the visibility is 16 kilometers (9 miles).', response_metadata={'token_usage': {'completion_tokens': 66, 'prompt_tokens': 1382, 'total_tokens': 1448}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_319be4768e', 'finish_reason': 'stop', 'logprobs': None}, id='run-88e6c5fc-fefe-4370-9556-616e7ab836c2-0')]}
```

比较提问的两个城市：

```
messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
```

输出内容：

```
{'messages': [AIMessage(content='Los Angeles is currently warmer than San Francisco. The temperature in Los Angeles is 16.7°C (62.1°F), while in San Francisco it is 12.8°C (55.0°F).', response_metadata={'token_usage': {'completion_tokens': 44, 'prompt_tokens': 1460, 'total_tokens': 1504}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_319be4768e', 'finish_reason': 'stop', 'logprobs': None}, id='run-8f4c50e6-0c2c-4e5d-a4bf-491d0bcb7b2b-0')]}
```

使用一个新的 thread_id，上下文丢失：

```
messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
```

输出

```
{'messages': [AIMessage(content='Your question is a bit ambiguous. Could you please clarify what you mean by "which one is warmer"? Are you comparing two specific locations, objects, or something else?', response_metadata={'token_usage': {'completion_tokens': 35, 'prompt_tokens': 149, 'total_tokens': 184}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_319be4768e', 'finish_reason': 'stop', 'logprobs': None}, id='run-30366ff0-961e-4181-a10c-feeef6543198-0')]}
```


## 流式输出

使用 astream_events

```
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver

memory = AsyncSqliteSaver.from_conn_string(":memory:")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)

messages = [HumanMessage(content="What is the weather in SF?")]
thread = {"configurable": {"thread_id": "4"}}

async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            # Empty content in the context of OpenAI means
            # that the model is asking for a tool to be invoked.
            # So we only print non-empty content
            print(content, end="|")
```

输出内容
```
Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_WkbSdRP4jOeNhF8HKYpg54d3'}
Back to the model!
The| current| weather| in| San| Francisco| is| over|cast| with| a| temperature| of| |12|.|8|°C| (|55|.|0|°F|).| The| wind| is| blowing| from| the| west|-s|outh|west| at| |16|.|1| mph| (|25|.|9| k|ph|),| and| the| humidity| is| at| |93|%.| The| feels|-like| temperature| is| about| |11|.|4|°C| (|52|.|4|°F|).|
```