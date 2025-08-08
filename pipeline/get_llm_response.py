from openai import OpenAI
from pathlib import Path
from data.models_desc import models
import creds
from promptconstructor.combine_txts import read_with_langchain
import os

MODEL = "qwen/qwen-2-72b-instruct"
PARENT_PATH = Path(os.path.dirname(__file__)).parent


def info(prompt, response, model_response):
    def cost(prompt, response):
        return len(prompt) / 1000 * models[MODEL]["in_price"] + \
               len(response) / 1000 * models[MODEL]["out_price"]

    system_prompt_price = len("You are a helpful assistant") / 1000 * models[MODEL]["in_price"]
    print(f"\n\nPrice: {cost(prompt, response):.5f}")
    print(f"Price with system prompt: {cost(prompt, response) + system_prompt_price:.5f}")
    print(f"Len(in_symbols): {len(prompt)}")
    print(f"Length of tokens, total: {model_response.usage.prompt_tokens + model_response.usage.completion_tokens}")
    print(f"Len(out_symbols): {len(response)}")


def get_debug_response(llm_iter=0, num=0):
    file_path = os.path.join(PARENT_PATH, "pipeline", "debug_llm_outputs", f"llm_iter_{llm_iter}",  f"out_{num}.txt")
    with open(file_path, encoding='utf-8') as debug:
        data = debug.read()
    return data


def get_response(file_name="continue-iter.txt", llm_iter=0, num=0, dir_name='burg', noise_level=0, print_info=False):
    # client = OpenAI(
    #     api_key=creds.api_key,
    #     base_url="https://api.vsegpt.ru/v1")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=creds.api_key_router,
    )

    if file_name != "zero-iter.txt":
        prompt_path = os.path.join(PARENT_PATH, "pipeline", "prompts", f"llm_iter_{llm_iter}", file_name)
    else:
        prompt_path = os.path.join(PARENT_PATH, "pipeline", "prompts", file_name)

    prompt = read_with_langchain(path=prompt_path, dir_name=dir_name, noise_level=noise_level) # 2446 base len
    messages = [{"role": "user", "content": prompt}]

    response_big = client.chat.completions.create(
        model="qwen/qwen-2.5-72b-instruct",
        messages=messages,
        temperature=1.0,
        n=1,
        max_tokens=2000, # максимальное число ВЫХОДНЫХ токенов
        extra_headers={ "X-Title": "EPDELLM"},)
    # response_big = client.chat.completions.create(
    #     model=MODEL,
    #     messages=messages,
    #     temperature=1.0,
    #     n=1,
    #     max_tokens=2000, # максимальное число ВЫХОДНЫХ токенов
    #     extra_headers={ "X-Title": "EPDELLM"},)

    response = response_big.choices[0].message.content
    if print_info:
        print("Response:", response)
        info(prompt, response, response_big)

    write_path = os.path.join(PARENT_PATH, "pipeline", "debug_llm_outputs", f"llm_iter_{llm_iter}",  f"out_{num}.txt")
    with open(write_path, 'w', encoding="utf-8") as model_out:
        model_out.write(response)
    return response


if __name__ == "__main__":
    get_response(print_info=True)