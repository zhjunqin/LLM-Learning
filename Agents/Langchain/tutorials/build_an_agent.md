# Build an Agent

单独的语言模型无法执行实际操作 - 它们只能输出文本。LangChain 的一个重要应用场景就是创建智能代理(agent)。智能代理利用语言模型作为推理引擎，来决定应该采取哪些行动，以及这些行动的输入应该是什么。这些行动的结果可以反馈给智能代理，让它判断是否需要采取更多行动，或者已经可以完成任务了。

在本教程中，我们将构建一个智能代理，它可以与多种不同的工具进行交互：一个是本地数据库，另一个是搜索引擎。您将能够向这个智能代理提出问题，观察它调用工具的过程，并与之进行对话。

## Concepts

将涵盖以下概念:
- 使用语言模型，特别是它们的工具调用能力
- 创建一个 Retriever 以向我们的 agent 公开特定信息
- 使用搜索工具在线查找内容
- 使用 LangGraph agent，它们使用 LLM 思考应该采取什么行动，并执行这些行动
- 使用 LangSmith 调试和跟踪应用程序

## Define tools

我们首先需要创建想要使用的工具。我们将使用两种工具：Tavily (用于在线搜索)以及一个我们将创建的本地索引检索器。

### Tavily

```
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(max_results=2)
search.invoke("what is the weather in SF")
```
```
[{'url': 'https://weather.com/weather/tenday/l/San Francisco CA USCA0987:1:US',
  'content': "Comfy & Cozy\nThat's Not What Was Expected\nOutside\n'No-Name Storms' In Florida\nGifts From On High\nWhat To Do For Wheezing\nSurviving The Season\nStay Safe\nAir Quality Index\nAir quality is considered satisfactory, and air pollution poses little or no risk.\n Health & Activities\nSeasonal Allergies and Pollen Count Forecast\nNo pollen detected in your area\nCold & Flu Forecast\nFlu risk is low in your area\nWe recognize our responsibility to use data and technology for good. recents\nSpecialty Forecasts\n10 Day Weather-San Francisco, CA\nToday\nMon 18 | Day\nConsiderable cloudiness. Tue 19\nTue 19 | Day\nLight rain early...then remaining cloudy with showers in the afternoon. Wed 27\nWed 27 | Day\nOvercast with rain showers at times."},
 {'url': 'https://www.accuweather.com/en/us/san-francisco/94103/hourly-weather-forecast/347629',
  'content': 'Hourly weather forecast in San Francisco, CA. Check current conditions in San Francisco, CA with radar, hourly, and more.'}]
```

### Retriever

我们还将创建一个自己数据的检索器。

```
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

inference_server_url = "https://open.bigmodel.cn/api/paas/v4"
embedding = OpenAIEmbeddings(model="embedding-2",
                             openai_api_key="xxx.xxx",
                             openai_api_base=inference_server_url,
                             check_embedding_ctx_length=False
                            )

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, embedding)
retriever = vector.as_retriever()

retriever.invoke("how to upload a dataset")[0]

```
```
Document(page_content='import Clientfrom langsmith.evaluation import evaluateclient = Client()# Define dataset: these are your test casesdataset_name = "Sample Dataset"dataset = client.create_dataset(dataset_name, description="A sample dataset in LangSmith.")client.create_examples(    inputs=[        {"postfix": "to LangSmith"},        {"postfix": "to Evaluations in LangSmith"},    ],    outputs=[        {"output": "Welcome to LangSmith"},        {"output": "Welcome to Evaluations in LangSmith"},    ],    dataset_id=dataset.id,)# Define your evaluatordef exact_match(run, example):    return {"score": run.outputs["output"] == example.outputs["output"]}experiment_results = evaluate(    lambda input: "Welcome " + input[\'postfix\'], # Your AI system goes here    data=dataset_name, # The data to predict and grade over    evaluators=[exact_match], # The evaluators to score the results    experiment_prefix="sample-experiment", # The name of the experiment    metadata={      "version": "1.0.0",      "revision_id":', metadata={'source': 'https://docs.smith.langchain.com/overview', 'title': 'Getting started with LangSmith | 🦜️🛠️ LangSmith', 'description': 'Introduction', 'language': 'en'})
```

现在我们已经填充了我们用于检索的索引，我们可以很容易地将其转换成一个工具(Agent需要适当使用的格式)。

```
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
```

### Tools

创建两个所需要的工具

```
tools = [search, retriever_tool]
```

## Using Language Models

接下来，让我们学习如何使用语言模型来调用工具。LangChain 支持许多不同的语言模型，您可以交替使用它们 - 选择您想要使用的模型!

```
from langchain_openai import ChatOpenAI

api_key = "xxx.xxx"
server_url = "https://open.bigmodel.cn/api/paas/v4"
model = ChatOpenAI(
    model="glm-3-turbo",
    openai_api_key=api_key,
    openai_api_base=server_url,
    # max_tokens=100,
    temperature=0.7,
)
```

你可以通过传入一个消息列表来调用语言模型。默认情况下，响应是一个内容字符串。

```
from langchain_core.messages import HumanMessage

response = model.invoke([HumanMessage(content="hi!")])
response.content
# 'Hello! How can I assist you today?'
```

现在我们可以看到如何让这个模型能够调用工具。为了实现这一点，我们使用 .bind_tools 来让语言模型获知这些工具。

```
model_with_tools = model.bind_tools(tools)
```

现在我们可以调用这个模型了。让我们先用一个普通的消息来调用它，看看它会如何响应。我们可以查看 content 字段和 tool_calls 字段。

```
response = model_with_tools.invoke([HumanMessage(content="Hi!")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

# ContentString: Hello! How can I assist you today?
# ToolCalls: []
```

现在，让我们尝试使用一些需要调用工具的输入来调用它。

```
response = model_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")
# ContentString: 
# ToolCalls: [{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in SF'}, 'id': 'call_nfE1XbCqZ8eJsB8rNdn4MQZQ'}]
```

我们可以看到，现在没有内容输出，但有一个工具调用！它希望我们调用 Tavily Search 工具。

这还没有实际调用该工具 - 它只是告诉我们需要调用它。为了真正调用它，我们需要创建我们的代理。

## Create the agent

现在我们已经定义了工具和 LLM，我们可以创建代理了。我们将使用 LangGraph 来构建代理。目前我们正在使用一个高级接口来构建代理，但 LangGraph 的好处是这个高级接口是基于一个低级、高度可控的 API，以防你想要修改代理的逻辑。

现在，我们可以用 LLM 和工具初始化 Agent 了。

请注意，我们传入的是 model，而不是 model_with_tools。这是因为 create_tool_calling_executor 会在幕后调用 .bind_tools 来完成这项工作。

```
from langgraph.prebuilt import chat_agent_executor

agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)
```

## Run the agent

现在我们可以对几个查询运行这个 Agent 了! 注意，目前这些都是无状态查询(它不会记住之前的交互)。注意，Agent 将在交互结束时返回最终状态(其中包括任何输入，我们稍后将看到如何只获取输出)。

首先，让我们看看当没有需要调用工具时，它如何响应：

```
response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})

response["messages"]
```
```
[HumanMessage(content='hi!', id='1535b889-10a5-45d0-a1e1-dd2e60d4bc04'),
 AIMessage(content='Hello! How can I assist you today?', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 129, 'total_tokens': 139}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-2c94c074-bdc9-4f01-8fd7-71cfc4777d55-0')]
```

可以到 LangSmith trace 上查看底层发生了什么。

接下来，我们尝试一个需要调用 retriever 的例子。

```
response = agent_executor.invoke(
    {"messages": [HumanMessage(content="how can langsmith help with testing?")]}
)
response["messages"]
```
```
[HumanMessage(content='how can langsmith help with testing?', id='04f4fe8f-391a-427c-88af-1fa064db304c'),
 AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_FNIgdO97wo51sKx3XZOGLHqT', 'function': {'arguments': '{\n  "query": "how can LangSmith help with testing"\n}', 'name': 'langsmith_search'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 135, 'total_tokens': 157}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-51f6ea92-84e1-43a5-b1f2-bc0c12d8613f-0', tool_calls=[{'name': 'langsmith_search', 'args': {'query': 'how can LangSmith help with testing'}, 'id': 'call_FNIgdO97wo51sKx3XZOGLHqT'}]),
 ToolMessage(content="Getting started with LangSmith | 🦜️🛠️ LangSmith\n\nSkip to main contentLangSmith API DocsSearchGo to AppQuick StartUser GuideTracingEvaluationProduction Monitoring & AutomationsPrompt HubProxyPricingSelf-HostingCookbookQuick StartOn this pageGetting started with LangSmithIntroduction\u200bLangSmith is a platform for building production-grade LLM applications. It allows you to closely monitor and evaluate your application, so you can ship quickly and with confidence. Use of LangChain is not necessary - LangSmith works on its own!Install LangSmith\u200bWe offer Python and Typescript SDKs for all your LangSmith needs.PythonTypeScriptpip install -U langsmithyarn add langchain langsmithCreate an API key\u200bTo create an API key head to the setting pages. Then click Create API Key.Setup your environment\u200bShellexport LANGCHAIN_TRACING_V2=trueexport LANGCHAIN_API_KEY=<your-api-key># The below examples use the OpenAI API, though it's not necessary in generalexport OPENAI_API_KEY=<your-openai-api-key>Log your first trace\u200bWe provide multiple ways to log traces\n\nLearn about the workflows LangSmith supports at each stage of the LLM application lifecycle.Pricing: Learn about the pricing model for LangSmith.Self-Hosting: Learn about self-hosting options for LangSmith.Proxy: Learn about the proxy capabilities of LangSmith.Tracing: Learn about the tracing capabilities of LangSmith.Evaluation: Learn about the evaluation capabilities of LangSmith.Prompt Hub Learn about the Prompt Hub, a prompt management tool built into LangSmith.Additional Resources\u200bLangSmith Cookbook: A collection of tutorials and end-to-end walkthroughs using LangSmith.LangChain Python: Docs for the Python LangChain library.LangChain Python API Reference: documentation to review the core APIs of LangChain.LangChain JS: Docs for the TypeScript LangChain libraryDiscord: Join us on our Discord to discuss all things LangChain!FAQ\u200bHow do I migrate projects between organizations?\u200bCurrently we do not support project migration betwen organizations. While you can manually imitate this by\n\nteam deals with sensitive data that cannot be logged. How can I ensure that only my team can access it?\u200bIf you are interested in a private deployment of LangSmith or if you need to self-host, please reach out to us at sales@langchain.dev. Self-hosting LangSmith requires an annual enterprise license that also comes with support and formalized access to the LangChain team.Was this page helpful?NextUser GuideIntroductionInstall LangSmithCreate an API keySetup your environmentLog your first traceCreate your first evaluationNext StepsAdditional ResourcesFAQHow do I migrate projects between organizations?Why aren't my runs aren't showing up in my project?My team deals with sensitive data that cannot be logged. How can I ensure that only my team can access it?CommunityDiscordTwitterGitHubDocs CodeLangSmith SDKPythonJS/TSMoreHomepageBlogLangChain Python DocsLangChain JS/TS DocsCopyright © 2024 LangChain, Inc.", name='langsmith_search', id='f286c7e7-6514-4621-ac60-e4079b37ebe2', tool_call_id='call_FNIgdO97wo51sKx3XZOGLHqT'),
 AIMessage(content="LangSmith is a platform that can significantly aid in testing by offering several features:\n\n1. **Tracing**: LangSmith provides robust tracing capabilities that enable you to monitor your application closely. This feature is particularly useful for tracking the behavior of your application and identifying any potential issues.\n\n2. **Evaluation**: LangSmith allows you to perform comprehensive evaluations of your application. This can help you assess the performance of your application under various conditions and make necessary adjustments to enhance its functionality.\n\n3. **Production Monitoring & Automations**: With LangSmith, you can keep a close eye on your application when it's in active use. The platform provides tools for automatic monitoring and managing routine tasks, helping to ensure your application runs smoothly.\n\n4. **Prompt Hub**: It's a prompt management tool built into LangSmith. This feature can be instrumental when testing various prompts in your application.\n\nOverall, LangSmith helps you build production-grade LLM applications with confidence, providing necessary tools for monitoring, evaluation, and automation.", response_metadata={'token_usage': {'completion_tokens': 200, 'prompt_tokens': 782, 'total_tokens': 982}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-4b80db7e-9a26-4043-8b6b-922f847f9c80-0')]
```

请注意，我们最终得到的状态也包含了工具调用和工具响应消息。

现在让我们尝试一个需要调用搜索工具的情况：

```
response = agent_executor.invoke(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
)
response["messages"]
```
```
[HumanMessage(content='whats the weather in sf?', id='e6b716e6-da57-41de-a227-fee281fda588'),
 AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_TGDKm0saxuGKJD5OYOXWRvLe', 'function': {'arguments': '{\n  "query": "current weather in San Francisco"\n}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 134, 'total_tokens': 157}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-fd7d5854-2eab-4fca-ad9e-b3de8d587614-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_TGDKm0saxuGKJD5OYOXWRvLe'}]),
 ToolMessage(content='[{"url": "https://www.weatherapi.com/", "content": "{\'location\': {\'name\': \'San Francisco\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 37.78, \'lon\': -122.42, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1714426800, \'localtime\': \'2024-04-29 14:40\'}, \'current\': {\'last_updated_epoch\': 1714426200, \'last_updated\': \'2024-04-29 14:30\', \'temp_c\': 17.8, \'temp_f\': 64.0, \'is_day\': 1, \'condition\': {\'text\': \'Sunny\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/day/113.png\', \'code\': 1000}, \'wind_mph\': 23.0, \'wind_kph\': 37.1, \'wind_degree\': 290, \'wind_dir\': \'WNW\', \'pressure_mb\': 1019.0, \'pressure_in\': 30.09, \'precip_mm\': 0.0, \'precip_in\': 0.0, \'humidity\': 50, \'cloud\': 0, \'feelslike_c\': 17.8, \'feelslike_f\': 64.0, \'vis_km\': 16.0, \'vis_miles\': 9.0, \'uv\': 5.0, \'gust_mph\': 27.5, \'gust_kph\': 44.3}}"}, {"url": "https://www.wunderground.com/hourly/us/ca/san-francisco/94125/date/2024-4-29", "content": "Current Weather for Popular Cities . San Francisco, CA warning 59 \\u00b0 F Mostly Cloudy; Manhattan, NY 56 \\u00b0 F Fair; Schiller Park, IL (60176) warning 58 \\u00b0 F Mostly Cloudy; Boston, MA 52 \\u00b0 F Sunny ..."}]', name='tavily_search_results_json', id='aa0d8c3d-23b5-425a-ad05-3c174fc04892', tool_call_id='call_TGDKm0saxuGKJD5OYOXWRvLe'),
 AIMessage(content='The current weather in San Francisco, California is sunny with a temperature of 64.0°F (17.8°C). The wind is coming from the WNW at a speed of 23.0 mph. The humidity level is at 50%. There is no precipitation and the cloud cover is 0%. The visibility is 16.0 km. The UV index is 5.0. Please note that this information is as of 14:30 on April 29, 2024, according to [Weather API](https://www.weatherapi.com/).', response_metadata={'token_usage': {'completion_tokens': 117, 'prompt_tokens': 620, 'total_tokens': 737}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-2359b41b-cab6-40c3-b6d9-7bdf7195a601-0')]
```

我们可以查看 LangSmith 跟踪记录，以确保它有效地调用了搜索工具。

## Streaming Messages

我们已经看到，可以使用 .invoke 调用代理来获得最终响应。如果代理正在执行多个步骤，这可能需要一定时间。为了显示中间进度，我们可以在发生时流式传回消息。

```
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
):
    print(chunk)
    print("----")
```

```
{'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_50Kb8zHmFqPYavQwF5TgcOH8', 'function': {'arguments': '{\n  "query": "current weather in San Francisco"\n}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 134, 'total_tokens': 157}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-042d5feb-c2cc-4c3f-b8fd-dbc22fd0bc07-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_50Kb8zHmFqPYavQwF5TgcOH8'}])]}}
----
{'action': {'messages': [ToolMessage(content='[{"url": "https://www.weatherapi.com/", "content": "{\'location\': {\'name\': \'San Francisco\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 37.78, \'lon\': -122.42, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1714426906, \'localtime\': \'2024-04-29 14:41\'}, \'current\': {\'last_updated_epoch\': 1714426200, \'last_updated\': \'2024-04-29 14:30\', \'temp_c\': 17.8, \'temp_f\': 64.0, \'is_day\': 1, \'condition\': {\'text\': \'Sunny\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/day/113.png\', \'code\': 1000}, \'wind_mph\': 23.0, \'wind_kph\': 37.1, \'wind_degree\': 290, \'wind_dir\': \'WNW\', \'pressure_mb\': 1019.0, \'pressure_in\': 30.09, \'precip_mm\': 0.0, \'precip_in\': 0.0, \'humidity\': 50, \'cloud\': 0, \'feelslike_c\': 17.8, \'feelslike_f\': 64.0, \'vis_km\': 16.0, \'vis_miles\': 9.0, \'uv\': 5.0, \'gust_mph\': 27.5, \'gust_kph\': 44.3}}"}, {"url": "https://world-weather.info/forecast/usa/san_francisco/april-2024/", "content": "Extended weather forecast in San Francisco. Hourly Week 10 days 14 days 30 days Year. Detailed \\u26a1 San Francisco Weather Forecast for April 2024 - day/night \\ud83c\\udf21\\ufe0f temperatures, precipitations - World-Weather.info."}]', name='tavily_search_results_json', id='d88320ac-3fe1-4f73-870a-3681f15f6982', tool_call_id='call_50Kb8zHmFqPYavQwF5TgcOH8')]}}
----
{'agent': {'messages': [AIMessage(content='The current weather in San Francisco, California is sunny with a temperature of 17.8°C (64.0°F). The wind is coming from the WNW at 23.0 mph. The humidity is at 50%. [source](https://www.weatherapi.com/)', response_metadata={'token_usage': {'completion_tokens': 58, 'prompt_tokens': 602, 'total_tokens': 660}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-0cd2a507-ded5-4601-afe3-3807400e9989-0')]}}
----
```

## Streaming tokens

除了流式传输消息之外，流式传输 tokens 也很有用。我们可以使用 .astream_events 方法来实现这一点。

```
async for event in agent_executor.astream_events(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}, version="v1"
):
    kind = event["event"]
    if kind == "on_chain_start":
        if (
            event["name"] == "Agent"
        ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
            print(
                f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
            )
    elif kind == "on_chain_end":
        if (
            event["name"] == "Agent"
        ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
            print()
            print("--")
            print(
                f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
            )
    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            # Empty content in the context of OpenAI means
            # that the model is asking for a tool to be invoked.
            # So we only print non-empty content
            print(content, end="|")
    elif kind == "on_tool_start":
        print("--")
        print(
            f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
        )
    elif kind == "on_tool_end":
        print(f"Done tool: {event['name']}")
        print(f"Tool output was: {event['data'].get('output')}")
        print("--")
```

```
--
Starting tool: tavily_search_results_json with inputs: {'query': 'current weather in San Francisco'}
Done tool: tavily_search_results_json
Tool output was: [{'url': 'https://www.weatherapi.com/', 'content': "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.78, 'lon': -122.42, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1714427052, 'localtime': '2024-04-29 14:44'}, 'current': {'last_updated_epoch': 1714426200, 'last_updated': '2024-04-29 14:30', 'temp_c': 17.8, 'temp_f': 64.0, 'is_day': 1, 'condition': {'text': 'Sunny', 'icon': '//cdn.weatherapi.com/weather/64x64/day/113.png', 'code': 1000}, 'wind_mph': 23.0, 'wind_kph': 37.1, 'wind_degree': 290, 'wind_dir': 'WNW', 'pressure_mb': 1019.0, 'pressure_in': 30.09, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 50, 'cloud': 0, 'feelslike_c': 17.8, 'feelslike_f': 64.0, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 5.0, 'gust_mph': 27.5, 'gust_kph': 44.3}}"}, {'url': 'https://www.weathertab.com/en/c/e/04/united-states/california/san-francisco/', 'content': 'San Francisco Weather Forecast for Apr 2024 - Risk of Rain Graph. Rain Risk Graph: Monthly Overview. Bar heights indicate rain risk percentages. Yellow bars mark low-risk days, while black and grey bars signal higher risks. Grey-yellow bars act as buffers, advising to keep at least one day clear from the riskier grey and black days, guiding ...'}]
--
The| current| weather| in| San| Francisco|,| California|,| USA| is| sunny| with| a| temperature| of| |17|.|8|°C| (|64|.|0|°F|).| The| wind| is| blowing| from| the| W|NW| at| a| speed| of| |37|.|1| k|ph| (|23|.|0| mph|).| The| humidity| level| is| at| |50|%.| [|Source|](|https|://|www|.weather|api|.com|/)|
```

## Adding in memory

正如之前提到的，这个代理是无状态的。这意味着它不会记住之前的交互。要给它添加记忆功能，我们需要传入一个检查点器(checkpointer)。在传入检查点器时，我们还必须在调用代理时传入一个 thread_id (以便它知道要从哪个线程/对话恢复)。


```
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")

agent_executor = chat_agent_executor.create_tool_calling_executor(
    model, tools, checkpointer=memory
)

config = {"configurable": {"thread_id": "abc123"}}

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob!")]}, config
):
    print(chunk)
    print("----")

```
```
{'agent': {'messages': [AIMessage(content='Hello Bob! How can I assist you today?', response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 131, 'total_tokens': 142}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-607733e3-4b8d-4137-ae66-8a4b8ccc8d40-0')]}}
----
```

```
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats my name?")]}, config
):
    print(chunk)
    print("----")
```
```
{'agent': {'messages': [AIMessage(content='Your name is Bob. How can I assist you further?', response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 154, 'total_tokens': 167}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-e1181ba6-732d-4564-b479-9f1ab6bf01f6-0')]}}
----
```

## Conclusion

到此为止了！在这个快速入门中，我们介绍了如何创建一个简单的代理。我们展示了如何流式传回响应 - 不仅是中间步骤，还有 tokens！我们还添加了内存，这样你就可以与它们进行对话。代理是一个复杂的话题，还有很多需要学习！

有关代理的更多信息，请查看 LangGraph 文档。那里有自己的一套概念、教程和操作指南。


## 参考文献
- https://python.langchain.com/v0.2/docs/tutorials/agents/