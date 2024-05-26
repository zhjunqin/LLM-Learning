import requests
import base64
import argparse
import json
import os
import time


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text',
                        type=str,
                        default="")
    return parser.parse_args()


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def non_streaming(args):
    print("\n")
    url = "http://127.0.0.1:8000/v2/models/ensemble/generate"

    query = args.text
    data = {
        "text_input": query,
        "max_tokens": 600,
    }

    # 发送 POST 请求
    start = time.time()
    response = requests.post(url, json=data)
    end = time.time()

    # 打印响应结果
    print("end - start", end - start)
    print(response.json()["text_output"])


def non_streaming(args):
    print("\n")
    args = parse_arguments()
    url = "http://127.0.0.1:8000/v2/models/ensemble/generate_stream"

    # 构建请求参数
    query = args.text
    data = {
        "text_input": query,
        "max_tokens": 500,
        "stream": True,
    }

    # 发送 POST 请求
    s = requests.Session()
    first_token = 0.0
    start_time = time.time()
    output_tokens = 0
    result = ""
    with s.post(url, json=data, stream=True) as response:
        # Iterate over the response in chunks
        for chunk in response.iter_content(chunk_size=None):
            print(chunk)
            chunk_bytes = chunk.strip()
            if chunk_bytes:
                chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")
                data = json.loads(chunk)
                print(data)
                result += data["text_output"]
                timestamp = time.time()
                if first_token == 0.0:
                    first_token = timestamp - start_time
                output_tokens += 1

    end_time = time.time()

    # 打印响应结果
    print(result)
    print("Time to first token", first_token)
    print("totoal_time", end_time - start_time)
    print("output_tokens", output_tokens)


if __name__ == '__main__':
    args = parse_arguments()
    non_streaming(args)
