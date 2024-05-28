# Vector stores and retrievers

这个教程将让您熟悉 LangChain 的向量存储和检索抽象。这些抽象旨在支持数据的检索--从(向量)数据库和其他来源--以便与 LLM 工作流集成。它们对于需要从数据源中提取数据进行推理的应用程序很重要，就像在检索增强生成(Retrieval-Augmented Generation，RAG)的情况下(请参见我们的 [RAG 教程](https://python.langchain.com/v0.2/docs/tutorials/rag/))。

这个指南主要关注文本数据的检索。我们将介绍以下概念:

- 文档(Documents)；
- 向量存储(Vector stores)；
- 检索器(Retrievers)。

## Documents

LangChain 实现了一个文档(Document)抽象，旨在表示一个文本单元及其相关元数据。它有两个属性:

- page_content: 表示内容的字符串;
- metadata: 包含任意元数据的字典。

metadata 属性可以捕获有关文档来源、与其他文档的关系以及其他信息。请注意，单个 Document 对象通常代表一个较大文档的一部分。

让我们生成一些示例文档:

```
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]
```
在这里，我们生成了五个文档，其中包含表示三个不同"来源"的元数据

## Vector stores

向量搜索是一种常见的存储和搜索非结构化数据(如非结构化文本)的方式。其思路是存储与文本相关的数值向量。给定一个查询，我们可以将其嵌入为相同维度的向量，并使用向量相似性度量来识别存储中的相关数据。

LangChain 的 VectorStore 对象包含用于向存储添加文本和 Document 对象以及使用各种相似性度量进行查询的方法。它们通常使用嵌入模型进行初始化，这些模型决定了文本数据如何转换为数值向量。

LangChain 包含与不同向量存储技术的集成套件。一些向量存储托管在供应商(如各种云供应商)处，需要特定的凭据才能使用；有些(如 Postgres)在可本地运行或通过第三方运行的独立基础设施上运行;其他一些可以在内存中运行，适用于轻量级工作负载。在这里，我们将使用 Chroma 演示 LangChain VectorStores 的使用情况，它包括一个内存实现。

为了实例化一个向量存储，我们通常需要提供一个嵌入模型来指定如何将文本转换为数值向量。在这里，我们将使用 OpenAI 嵌入。


```
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

inference_server_url = "https://open.bigmodel.cn/api/paas/v4"
embedding = OpenAIEmbeddings(model="embedding-2",
                             openai_api_key="xxx.xxx",
                             openai_api_base=inference_server_url,
                             check_embedding_ctx_length=False
                            )
vectorstore = Chroma.from_documents(
    documents,
    embedding=embedding,
)
```

在此调用 .from_documents 将文档添加到向量存储中。VectorStore 实现了添加文档的方法，也可以在实例化对象后调用。大多数实现将允许您连接到现有的向量存储 - 例如，通过提供客户端、索引名称或其他信息。有关特定集成的更多详细信息，请查看[文档](https://python.langchain.com/v0.2/docs/integrations/vectorstores/)。

一旦我们实例化了包含文档的 VectorStore，我们就可以对其进行查询。VectorStore 包括以下查询方法:

- 同步和异步；
- 通过字符串查询和向量；
- 带有和不带有返回相似度分数；
- 通过相似度和 [maximum marginal relevance](https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.VectorStore.html#langchain_core.vectorstores.VectorStore.max_marginal_relevance_search)(平衡检索结果的相似性和多样性)。

这些方法的输出通常包括一个 Document 对象列表。

同步检索
```
vectorstore.similarity_search("cat")
```
```
[Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Dogs are great companions, known for their loyalty and friendliness.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Rabbits are social animals that need plenty of space to hop around.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Parrots are intelligent birds capable of mimicking human speech.', metadata={'source': 'bird-pets-doc'})]
```

异步检索
```
await vectorstore.asimilarity_search("cat")
```
```
[Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Dogs are great companions, known for their loyalty and friendliness.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Rabbits are social animals that need plenty of space to hop around.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Parrots are intelligent birds capable of mimicking human speech.', metadata={'source': 'bird-pets-doc'})]
```

返回的结果带分数
```
# 请注意，不同的提供商实现了不同的评分机制;在这里，Chroma 返回的是一个距离度量指标，它应该与相似性成反比。
vectorstore.similarity_search_with_score("cat")
```
```
[(Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'}),
  0.3751849830150604),
 (Document(page_content='Dogs are great companions, known for their loyalty and friendliness.', metadata={'source': 'mammal-pets-doc'}),
  0.48316916823387146),
 (Document(page_content='Rabbits are social animals that need plenty of space to hop around.', metadata={'source': 'mammal-pets-doc'}),
  0.49601367115974426),
 (Document(page_content='Parrots are intelligent birds capable of mimicking human speech.', metadata={'source': 'bird-pets-doc'}),
  0.4972994923591614)]
```

使用 Embedding 向量直接查询

```
embedding = OpenAIEmbeddings().embed_query("cat")

vectorstore.similarity_search_by_vector(embedding)
```
```
[Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Dogs are great companions, known for their loyalty and friendliness.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Rabbits are social animals that need plenty of space to hop around.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Parrots are intelligent birds capable of mimicking human speech.', metadata={'source': 'bird-pets-doc'})]
```

## Retrievers

LangChain 的 VectorStore 对象并不是 Runnable 的子类，因此无法立即集成到 LangChain 表达式语言(LangChain Expression Language, LCEL)链中。

LangChain 的 Retrievers 是 Runnables，因此它们实现了一组标准的方法(如同步和异步 invoke 和批处理操作)，并且被设计用于并入 LCEL 链中。

我们可以自己创建一个简单版本，而不需要继承 Retriever。如果我们选择使用哪种方法来检索文档，我们就可以轻松创建一个 Runnable。下面我们将围绕 similarity_search 方法构建一个:

```
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

# vectorstore.similarity_search 是一个 method
retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result

retriever.batch(["cat", "shark"])
```

```
[[Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'})],
 [Document(page_content='Goldfish are popular pets for beginners, requiring relatively simple care.', metadata={'source': 'fish-pets-doc'})]]
```

VectorStore 实现了一个 as_retriever 方法，该方法将生成一个 Retriever，具体来说是一个 VectorStoreRetriever。这些 Retriever 包括特定的 search_type 和 search_kwargs 属性，用于标识要调用底层向量存储的哪些方法，以及如何对它们进行参数化。例如，我们可以使用以下方式来复制上述情况:

```
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

retriever.batch(["cat", "shark"])
```

```
[[Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'})],
 [Document(page_content='Goldfish are popular pets for beginners, requiring relatively simple care.', metadata={'source': 'fish-pets-doc'})]]
```

VectorStoreRetriever 支持以下三种搜索类型："similarity"(默认)、"mmr"(最大边际相关性，如上所述)以及"similarity_score_threshold"。我们可以使用后者来根据相似度分数对 Retriever 的输出进行阈值过滤。

Retrievers 可以很容易地并入到更复杂的应用程序中，例如检索增强型生成(RAG)应用程序，它将给定的问题与检索到的上下文结合成为输入到 LLM 的提示。下面我们展示一个最简单的示例。

```
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

api_key = "xxx.xxx"
server_url = "https://open.bigmodel.cn/api/paas/v4"
llm = ChatOpenAI(
    model="glm-3-turbo",
    openai_api_key=api_key,
    openai_api_base=server_url,
    max_tokens=100,
    temperature=0.7,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

response = rag_chain.invoke("tell me about cats")

print(response.content) # Cats are independent pets that often enjoy their own space.

```

## Learn more

检索策略可以非常丰富和复杂。例如:

- 可以从查询中[推断出硬性规则和过滤器](https://python.langchain.com/v0.2/docs/how_to/self_query/)(例如，"使用2020年后发布的文档")；
- 可以返回以某种方式与检索到的[上下文相关联的文档](https://python.langchain.com/v0.2/docs/how_to/parent_document_retriever/)(例如，通过某种文档分类体系)；
- 可以为每个上下文单元生成[多个嵌入](https://python.langchain.com/v0.2/docs/how_to/multi_vector/)；
- 可以[集成来自多个检索器](https://python.langchain.com/v0.2/docs/how_to/ensemble_retriever/)的结果；
- 可以为文档分配权重，例如提高[近期文档的权重](https://python.langchain.com/v0.2/docs/how_to/time_weighted_vectorstore/)。

指南中的[检索器部分](https://python.langchain.com/v0.2/docs/how_to/#retrievers)涵盖了这些内置检索策略以及其他策略。

此外，也可以很容易地扩展 [BaseRetriever](https://api.python.langchain.com/en/latest/retrievers/langchain_core.retrievers.BaseRetriever.html) 类来实现自定义检索器。请参阅我们的[指南](https://python.langchain.com/v0.2/docs/how_to/custom_retriever/)。


## 参考文献
- https://python.langchain.com/v0.2/docs/tutorials/retrievers/