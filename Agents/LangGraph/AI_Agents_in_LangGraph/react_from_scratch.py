import openai
import re
import httpx
import os
import json
from openai import OpenAI

base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key = "xxx.xxx"
model_name = "glm-4-air"

client = OpenAI(api_key=api_key, base_url=base_url)

chat_completion = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": "Hello world"}]
)

print(chat_completion.choices[0].message.content)


class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model="glm-4-air",
            temperature=0.1,
            messages=self.messages)
        return completion.choices[0].message.content


prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()


def calculate(what):
    return eval(what)


def average_dog_weight(name):
    if name in "Scottish Terrier":
        return ("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return ("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return ("a toy poodles average weight is 7 lbs")
    else:
        return ("An average dog weights 50 lbs")


known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}


abot = Agent(prompt)
result = abot("How much does a toy poodle weigh?")
print(result)
"""
Thought: I should find out the average weight of a toy poodle using the average_dog_weight function.
Action: average_dog_weight: Toy Poodle
PAUSE
"""

result = average_dog_weight("Toy Poodle")
next_prompt = "Observation: {}".format(result)
abot(next_prompt)
print(json.dumps(abot.messages, indent=4))
"""
[
    {
        "role": "system",
        "content": "You run in a loop of Thought, Action, PAUSE, Observation.\nAt the end of the loop you output an Answer\nUse Thought to describe your thoughts about the question you have been asked.\nUse Action to run one of the actions available to you - then return PAUSE.\nObservation will be the result of running those actions.\n\nYour available actions are:\n\ncalculate:\ne.g. calculate: 4 * 7 / 3\nRuns a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary\n\naverage_dog_weight:\ne.g. average_dog_weight: Collie\nreturns average weight of a dog when given the breed\n\nExample session:\n\nQuestion: How much does a Bulldog weigh?\nThought: I should look the dogs weight using average_dog_weight\nAction: average_dog_weight: Bulldog\nPAUSE\n\nYou will be called again with this:\n\nObservation: A Bulldog weights 51 lbs\n\nYou then output:\n\nAnswer: A bulldog weights 51 lbs"
    },
    {
        "role": "user",
        "content": "How much does a toy poodle weigh?"
    },
    {
        "role": "assistant",
        "content": "Thought: I should find out the average weight of a toy poodle using the average_dog_weight function.\nAction: average_dog_weight: Toy Poodle\nPAUSE"
    },
    {
        "role": "user",
        "content": "Observation: a toy poodles average weight is 7 lbs"
    },
    {
        "role": "assistant",
        "content": "Answer: A toy poodle weighs an average of 7 lbs."
    }
]
"""


abot = Agent(prompt)
question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
abot(question)

next_prompt = "Observation: {}".format(average_dog_weight("Border Collie"))
print(next_prompt)

abot(next_prompt)
next_prompt = "Observation: {}".format(average_dog_weight("Scottish Terrier"))
print(next_prompt)

abot(next_prompt)
next_prompt = "Observation: {}".format(eval("37 + 20"))
print(next_prompt)

abot(next_prompt)
print(json.dumps(abot.messages, indent=4))
"""
[
    {
        "role": "system",
        "content": "You run in a loop of Thought, Action, PAUSE, Observation.\nAt the end of the loop you output an Answer\nUse Thought to describe your thoughts about the question you have been asked.\nUse Action to run one of the actions available to you - then return PAUSE.\nObservation will be the result of running those actions.\n\nYour available actions are:\n\ncalculate:\ne.g. calculate: 4 * 7 / 3\nRuns a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary\n\naverage_dog_weight:\ne.g. average_dog_weight: Collie\nreturns average weight of a dog when given the breed\n\nExample session:\n\nQuestion: How much does a Bulldog weigh?\nThought: I should look the dogs weight using average_dog_weight\nAction: average_dog_weight: Bulldog\nPAUSE\n\nYou will be called again with this:\n\nObservation: A Bulldog weights 51 lbs\n\nYou then output:\n\nAnswer: A bulldog weights 51 lbs"
    },
    {
        "role": "user",
        "content": "I have 2 dogs, a border collie and a scottish terrier. What is their combined weight"
    },
    {
        "role": "assistant",
        "content": "Thought: I need to find the average weight of a border collie and a scottish terrier, then add them together to get the combined weight.\n\nAction: average_dog_weight: Border Collie\nPAUSE"
    },
    {
        "role": "user",
        "content": "Observation: a Border Collies average weight is 37 lbs"
    },
    {
        "role": "assistant",
        "content": "Thought: Now I need to find the average weight of a Scottish Terrier.\n\nAction: average_dog_weight: Scottish Terrier\nPAUSE"
    },
    {
        "role": "user",
        "content": "Observation: Scottish Terriers average 20 lbs"
    },
    {
        "role": "assistant",
        "content": "Thought: Now that I have the average weights of both a Border Collie and a Scottish Terrier, I can add them together to find the combined weight.\n\nAction: calculate: 37 + 20\nPAUSE"
    },
    {
        "role": "user",
        "content": "Observation: 57"
    },
    {
        "role": "assistant",
        "content": "Answer: The combined weight of a Border Collie and a Scottish Terrier is 57 lbs."
    }
]
"""
