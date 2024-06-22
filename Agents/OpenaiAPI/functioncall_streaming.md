# Function Streaming Demo

参考 https://community.openai.com/t/help-for-function-calls-with-streaming/627170/6


## 示例

```
from openai import OpenAI
import openai
import json
# from apikey import api_key  # Ensure this imports your API key correctly, or if you wanna do it my way, make a file named apikey.py and put api_key="REAL_API_KEY_HERE" in it, and put that in the same folder as this file.

# openai.api_key = api_key

api_key = "sk-xxx"
base_url = "xxx"

client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
    base_url=base_url
)

# Hard-coded 'fake' weather API, in a real situation, this would be a real weather api, or anything else.
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    print(f"PRINT STATEMENT: Getting weather for {location} with unit {unit}") #Clairfication that the function actually is running and the model isn't making stuff up.
    print() # empty print statement to add space and seperate from API response
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "32", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

    
def run_conversation():
    messages = [{
        "role": "user",
        "content": "first, define unicorn in 30 words. Then find the weather in Paris"
    }]

    # We define the tool array here instead of within the API call because it's just eaiser to look at and manage, just like with messages up there ^.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g., San Francisco, CA"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of measurement for the temperature", "default": "fahrenheit"}
                    },
                    "required": ["location"]
                },
            }
        }
    ]

    print("======")
    print("messages: ", messages)
    stream = client.chat.completions.create(
        model="gpt-4-0125-preview", # the model you wanna use, if this doesn't work, try using "gpt-3.5-turbo-0125"
        messages=messages, # define the context, ususally this would be a thing that isn't static
        tools=tools, # the array of tools we defined up there ^
        tool_choice="auto", # pretty sure this is default, so you don't need it, but it's here just in case.
        stream=True, # enable streaming for the API
    )

    available_functions = {"get_current_weather": get_current_weather,
                        # "add_another_function_here": add_another_function_here,
                           }

    tool_call_accumulator = ""  # Accumulator for JSON fragments of tool call arguments
    tool_call_id = None  # Current tool call ID

    # This is where we print the chunks directly from the API if no function was called from the model
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True) # the extra stuff at the end makes it so it updates as fast as possible, and doesn't create new lines for each chunk it gets

        if chunk.choices[0].delta.tool_calls:
            for tc in chunk.choices[0].delta.tool_calls:
                if tc.id:  # New tool call detected here
                    tool_call_id = tc.id
                tool_call_accumulator += tc.function.arguments if tc.function.arguments else ""

                # When the accumulated JSON string seems complete then:
                try:
                    func_args = json.loads(tool_call_accumulator)
                    function_name = tc.function.name if tc.function.name else "get_current_weather"
                    # Call the corresponding function that we defined and matches what is in the available functions
                    func_response = json.dumps(available_functions[function_name](**func_args))
                    # Append the function response directly to messages
                    messages.append({
                        "tool_call_id": tool_call_id,
                        "role": "function",
                        "name": function_name,
                        "content": func_response,
                    })
                    tool_call_accumulator = ""  # Reset for the next tool call
                except json.JSONDecodeError:
                    # Incomplete JSON; continue accumulating
                    pass

    # Make a follow-up API call with the updated messages, including function call responses with tool id
    print("======")
    print("messages: ", messages)
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
    )

    # Prints each chunk as they come after the function is called and the result is available.
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)

run_conversation()

```
输出

```
======
messages:  [{'role': 'user', 'content': 'first, define unicorn in 30 words. Then find the weather in Paris'}]
A unicorn is a mythical creature represented as a horse with a single horn projecting from its forehead. Often associated with purity and grace, it is a symbol of beauty.

Okay, now let's check the weather in Paris.
ChoiceDeltaToolCall(index=0, id='call_hLsrAMmqbaqdGJOfznCqbLKq', function=ChoiceDeltaToolCallFunction(arguments='', name='get_current_weather'), type='function')
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='{\n', name=None), type=None)
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments=' ', name=None), type=None)
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments=' "', name=None), type=None)
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='location', name=None), type=None)
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='":', name=None), type=None)
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments=' "', name=None), type=None)
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='Paris', name=None), type=None)
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='"\n', name=None), type=None)
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='}', name=None), type=None)
PRINT STATEMENT: Getting weather for Paris with unit fahrenheit

======
messages:  [{'role': 'user', 'content': 'first, define unicorn in 30 words. Then find the weather in Paris'}, {'tool_call_id': 'call_hLsrAMmqbaqdGJOfznCqbLKq', 'role': 'function', 'name': 'get_current_weather', 'content': '"{\\"location\\": \\"Paris\\", \\"temperature\\": \\"22\\", \\"unit\\": \\"fahrenheit\\"}"'}]
A unicorn is a mythical creature resembling a horse with a single, spiraled horn on its forehead, symbolizing purity, magic, and rarity in various cultural legends and folklore.

As for the weather in Paris, it is currently 22 degrees Fahrenheit.
```

function call 是 arguments 不断在输出。