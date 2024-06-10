# A simple Python implementation of the ReAct pattern for LLMs

来源：https://til.simonwillison.net/llms/python-react-pattern

这段文本描述了一个关于人工智能的"梦魇场景"，即赋予人工智能访问工具的能力，让其能够调用API、执行自己的代码，从而摆脱其初始环境的限制。

它介绍了一种称为"ReAct模式"(Reason+Act)的技术，该模式允许人工智能模型执行额外的操作，如搜索维基百科或进行计算，并教会它如何请求执行这些操作，然后将结果反馈给模型。

文章还提到了一篇博客文章，探讨了以一定的成本($85,000)在浏览器中训练一个超越ChatGPT的模型的可能性。

另一篇文章则描述了这种模式的"surprising ease and effectiveness"，并指出ChatGPT价格的1/10下降使其成为一个理想的候选者来实现这个模式。该作者使用了langchain库来实现这个模式。

最后，文本给出了一个使用作者自己的小型 Python 包装器为 ChatGPT API 实现这个模式的初步尝试，该模式提供了三种新的操作:搜索维基百科、搜索作者的博客以及使用Python的eval()函数进行计算(这被认为是非常危险的，应该改用WebAssembly沙箱等更安全的方式)。