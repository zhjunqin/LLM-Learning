# Build a Chatbot

我们将介绍一个如何设计和实现基于 LLM 的聊天机器人的例子。这个聊天机器人将能够进行对话并记住之前的互动。

请注意，我们构建的这个聊天机器人只会使用语言模型来进行对话。还有几个相关的概念你可能会感兴趣:

- [Conversational RAG](https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/): 在外部数据源上提供聊天机器人
- [Agents](https://python.langchain.com/v0.2/docs/tutorials/agents/): 构建一个可以采取行动的聊天机器人

本教程将介绍基础知识，这对于这两个更高级的主题都会很有帮助。不过如果你愿意的话，也可以直接跳到那些主题。

## Concepts

我们将使用以下几个高级组件来开发聊天机器人：

- 聊天模型(Chat Models)。聊天机器人的接口是基于消息而不是原始文本，因此更适合使用聊天模型而不是文本语言模型(LLM)。
- 提示模板(Prompt Templates)，它简化了组装提示的过程，提示可以包括默认消息、用户输入、聊天历史记录以及（可选的）额外检索的上下文。
- 聊天历史记录(Chat History)，它允许聊天机器人"记住"过去的互动，并在回答后续问题时考虑这些历史记录。
- 使用 LangSmith 对应用程序进行调试和追踪。

我们将介绍如何将上述组件组合在一起，创建一个强大的对话式聊天机器人。

## Quickstart

将聊天历史全部送到模型里面去，可以让 LLM 知道过去的记录。

```
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

api_key = "xxx.xxx"
server_url = "https://open.bigmodel.cn/api/paas/v4"

model = ChatOpenAI(
    model="glm-3-turbo",
    openai_api_key=api_key,
    openai_api_base=server_url,
    max_tokens=100,
    temperature=0.7,
)

model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)

# AIMessage(content='According to your message, your name is Bob. How can I help you today?', response_metadata={'token_usage': {'completion_tokens': 20, 'prompt_tokens': 30, 'total_tokens': 50}, 'model_name': 'glm-3-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-a15c4175-442d-4915-8e8a-2571119ddc3f-0')

```

## Message History

我们可以使用一个消息历史记录(Message History)类来封装我们的模型，使其具有状态性。这将跟踪模型的输入和输出，并将它们存储在某个数据存储中。未来的交互将会加载这些消息，并将它们作为输入的一部分传递给链条。让我们看看如何使用这个功能！

首先，我们需要确保安装 langchain-community，因为我们将使用它提供的一个集成来存储消息历史记录。

```
# ! pip install langchain_community
```

安装完 langchain-community 之后，我们可以导入相关的类，并设置一个包含模型和消息历史记录的 Chain。这里很关键的是我们传递给 get_session_history 函数的函数。这个函数需要接受一个 session_id 参数，并返回一个消息历史记录(Message History)对象。这个 session_id 用于区分不同的对话，在调用新的 Chain 时应该作为配置的一部分传递进去(我们稍后会演示如何操作)。

```
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(model, get_session_history)

# 配置 session_id
config = {"configurable": {"session_id": "abc2"}}

response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Bob")],
    config=config,
)
response.content # 'Hello Bob! How can I assist you today?'

response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
response.content # 'Your name is Bob.'

# 修改成不同的 session_id
config = {"configurable": {"session_id": "abc3"}}

response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
response.content # "I'm sorry, I do not have the ability to know your name unless you tell me."

# 使用之前的 session_id
config = {"configurable": {"session_id": "abc2"}}

response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
response.content # 'Your name is Bob.'

```


## Prompt templates

提示模板(Prompt Templates)有助于将原始用户信息转换成语言模型(LLM)可以处理的格式。在这种情况下，原始用户输入只是一条消息，我们将其传递给 LLM。现在让我们让它更复杂一些。首先，我们将添加一条系统消息，其中包含一些自定义指令(但仍然将消息作为输入)。接下来，我们将添加更多的输入，而不仅仅是消息。

首先，让我们添加一条 system 消息。为此，我们将创建一个 ChatPromptTemplate。我们将使用 MessagesPlaceholder 来传递所有的消息。

```
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

response = chain.invoke({"messages": [HumanMessage(content="hi! I'm bob")]})

response.content # 'Hello, Bob! How can I assist you today?'

# 我们现在可以将其包装在与之前相同的消息历史记录(Messages History)对象中。
with_message_history = RunnableWithMessageHistory(chain, get_session_history)
config = {"configurable": {"session_id": "abc5"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Jim")],
    config=config,
)
response.content # 'Hello, Jim! How can I assist you today?'

# 使用 history 继续提问
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)

response.content # 'Your name is Bob.'
```

现在让我们让我们的提示更加复杂一些。假设提示模板现在看起来是这样的：
```
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = prompt | model

response = chain.invoke(
    {"messages": [HumanMessage(content="hi! I'm bob")], "language": "Spanish"}
)

response.content # '¡Hola Bob! ¿En qué puedo ayudarte hoy?'

# 使用 message history
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)
config = {"configurable": {"session_id": "abc11"}}

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="hi! I'm todd")], "language": "Spanish"},
    config=config,
)
response.content # '¡Hola Todd! ¿En qué puedo ayudarte hoy?'

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="whats my name?")], "language": "Spanish"},
    config=config,
)
response.content # 'Tu nombre es Todd. ¿Hay algo más en lo que pueda ayudarte?'

```

## Managing Conversation History

在构建聊天机器人时，理解如何管理对话历史记录是一个重要的概念。如果不加管理，消息列表将无限增长，可能会溢出语言模型的上下文窗口。因此，在传递消息之前，添加一个步骤来限制消息的大小是非常重要的。

重要的是，您需要在加载之前的消息历史记录之后，但在使用提示模板之前执行这个步骤。

我们可以通过在提示前添加一个简单的步骤来修改 messages 键，然后将这个新的 Chain 包装在消息历史记录(Message History)类中。首先，让我们定义一个函数来修改传入的消息。我们可以让它选择最近的 k 条消息。然后我们可以创建一个新的 Chain，将这个函数添加到 Chain 的开头。

```
from langchain_core.runnables import RunnablePassthrough

def filter_messages(messages, k=10):
    return messages[-k:]

chain = (
    RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
    | prompt
    | model
)

messages = [
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# 消息历史中第一条超过了 10 条
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "English",
    }
)
response.content # "I'm sorry, I don’t have access to your name. Can I help you with anything else?"

# 提问的信息在 10 条消息历史内
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my fav ice cream")],
        "language": "English",
    }
)
response.content # 'You mentioned that you like vanilla ice cream.'


# 使用 message history 封装
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc20"}}

response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
)
response.content # "I'm sorry, I don't know your name."

# 消息历史中增加了两条消息，关于 favorite ice cream 消息超过了历史
response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="whats my favorite ice cream?")],
        "language": "English",
    },
    config=config,
)

response.content # "I'm sorry, I don't know your favorite ice cream flavor."
```


## Streaming

现在我们已经有了一个聊天机器人功能。然而，聊天机器人应用程序的一个非常重要的用户体验考虑因素是流式传输(streaming)。语言模型有时需要一些时间来响应，为了改善用户体验，大多数应用程序都会以流式传输的方式逐个返回每个生成的令牌(token)。这样可以让用户看到处理的进度。

这实际上非常容易实现！

所有的 chain 都公开了一个 .stream 方法，使用消息历史记录的 chain 也不例外。我们可以简单地使用这个方法来获得流式传输的响应。

```
config = {"configurable": {"session_id": "abc15"}}
for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
        "language": "English",
    },
    config=config,
):
    print(r.content, end="|")
```

```
|Sure|,| Todd|!| Here|'s| a| joke| for| you|:

|Why| don|'t| scientists| trust| atoms|?

|Because| they| make| up| everything|!||
```

## Next Steps

现在您已经了解了如何在 LangChain 中创建聊天机器人的基础知识，一些更高级的教程可能会让您感兴趣：

- [对话性 RAG(Conversational RAG)](https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/)：在外部数据源上启用聊天机器人体验
- [代理(Agents)](https://python.langchain.com/v0.2/docs/tutorials/agents/)：构建一个可以执行 Action 的聊天机器人

如果您想深入了解一些具体的内容，值得查看的一些内容包括:

- [流式传输(Streaming)](https://python.langchain.com/v0.2/docs/how_to/streaming/)：流式传输对聊天应用程序至关重要
- [如何添加消息历史记录](https://python.langchain.com/v0.2/docs/how_to/message_history/)：对消息历史记录的所有相关内容进行更深入的探讨

## 参考文献
- https://python.langchain.com/v0.2/docs/tutorials/chatbot/
