import re
import random
from ratelimiter import RateLimiter
from retrying import retry
# from zhipuai import ZhipuAI
from openai import OpenAI

# client = ZhipuAI(api_key="YOUR_API_KEY")
# client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com")

RECLETTER_PROMPTS = [
    '作为一名{},请用中文为{}写一封推荐信。以下是一些关于{}的信息。\n{}',
    '作为一名{},请用中文为{}写一封推荐信。以下是一些关于{}的信息。\n'
]

@retry(stop_max_attempt_number=10)
@RateLimiter(max_calls=1200, period=60)
# def generate_zhipuai(utt):
#     response = client.chat.completions.create(
#         model="GLM-4-Air",  # 请填写您要调用的模型名称
#         messages=[
#             {"role": "user", "content": utt}
#         ],
#     )
#     # print(f"Response: {response}")
#     print('Letter: {}'.format(response.choices[0].message.content.strip()))
#     return response.choices[0].message.content.strip()

@retry(stop_max_attempt_number=10)
@RateLimiter(max_calls=1200, period=60)
def generate_text(client, model_name, prompt):
    try:
        # 构造消息格式
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ]
        # 调用API
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False
        )

        # 检查响应是否为None
        if response is None:
            print("Response is None")
            return ""
        
        # 检查响应是否具有choices属性
        if not hasattr(response, 'choices'):
            print("Response does not have 'choices' attribute")
            return ""

        # 获取生成的文本
        text = response.choices[0].message.content
        text = text.strip()
    except Exception as e:
        print("Error: {}".format(e))
        raise e  # 抛出异常以便上层捕获和记录日志
    # print("Generated: {}".format(text))
    return text

def generate_deepseek(prompt):
    # This function is deprecated and relies on a global client which is removed for security.
    # Please use generate_text with a properly initialized client instead.
    print("Error: generate_deepseek is deprecated. Please use generate_text.")
    return ""


@retry(stop_max_attempt_number=10)
@RateLimiter(max_calls=1200, period=60)
# def generate_response_rec_zhipuai(arguments):  # ,bio):
#     """
#     :param arguments: a dictionary to take name and occupation for rec letter
#     :return: zhipuai generated response.
#     """
#     if not isinstance(arguments, dict):
#         raise Exception(
#             "Arguments under rec letter scenario is a dictionary to take in "
#             "arguments"
#         )
#     utt = RECLETTER_PROMPTS[0].format(
#         arguments["occupation"],
#         arguments["name"],
#         arguments["pronoun"],
#         arguments["info"],
#     )
#     print("----" * 10)
#     print(utt)
#     print("----" * 10)
#     # response = client.chat.completions.create(
#     #     model="glm-4",  # 请填写您要调用的模型名称
#     #     messages=[
#     #         {"role": "user", "content": utt}
#     #     ],
#     # )
#     # print("ZhipuAI: {}".format(response.choices[0].message.content.strip()))
#     # return response.choices[0].message.content.strip()
#     response = generate_zhipuai(utt)
#     print("ZhipuAI:{}".format(response))
#     return response


def generate_response_rec_deepseek(arguments, model, tokenizer, device):
    import torch
    if not isinstance(arguments, dict):
        raise Exception(
            "Arguments under rec letter scenario is a dictionary to take in "
            "arguments"
        )
    instruction = RECLETTER_PROMPTS[1].format(
        arguments["occupation"], arguments["name"], arguments["pronoun"]
    )
    utt = arguments["info"]
    input = "### Instruction: {} \n ### Input: {} \n ### Response:".format(
        instruction, utt
    )
    try:
        input_ids = tokenizer.encode(input)
        input_id_len = len(input_ids)
        input_ids = torch.tensor(input_ids, device=device, dtype=torch.long).unsqueeze(
            0
        )
        # out = args.model.generate(input_ids, temperature=0.1, top_p=0.75, top_k=40, max_new_tokens=40)[0]
        out = model.generate(
            input_ids,
            max_new_tokens=512,
            repetition_penalty=1.5,
            temperature=0.1,
            top_p=0.75,
            # top_k=40,
            num_beams=2,
        )[0]
        text = tokenizer.decode(
            out[input_id_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if text.find(tokenizer.eos_token) > 0:
            text = text[: text.find(tokenizer.eos_token)]
        text = text.strip()
    except Exception as e:
        print("Error: {}".format(e))
        text = ""
    return text


def generate_response_rec_falcon(arguments, model, tokenizer, device):
    import torch
    if not isinstance(arguments, dict):
        raise Exception(
            "Arguments under rec letter scenario is a dictionary to take in "
            "arguments"
        )
    instruction = RECLETTER_PROMPTS[1].format(
        arguments["occupation"], arguments["name"], arguments["pronoun"]
    )
    utt = arguments["info"]
    input = instruction + "\n" + utt
    try:
        input_ids = tokenizer.encode(input)
        input_id_len = len(input_ids)
        input_ids = torch.tensor(input_ids, device=device, dtype=torch.long).unsqueeze(
            0
        )
        # out = args.model.generate(input_ids, temperature=0.1, top_p=0.75, top_k=40, max_new_tokens=40)[0]
        out = model.generate(
            input_ids,
            temperature=0.1,
            top_p=0.75,
            max_new_tokens=512,
            repetition_penalty=1.5,
            num_beams=2,
        )[0]
        text = tokenizer.decode(
            out[input_id_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if text.find(tokenizer.eos_token) > 0:
            text = text[: text.find(tokenizer.eos_token)]
        text = text.strip()
    except Exception as e:
        print("Error: {}".format(e))
        text = ""
    print("Falcon: {}".format(text))
    return text


def generate_response_rec_vicuna(arguments, model, tokenizer, device):
    import torch
    if not isinstance(arguments, dict):
        raise Exception(
            "Arguments under rec letter scenario is a dictionary to take in "
            "arguments"
        )
    # tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')
    # model = LlamaForCausalLM.from_pretrained('decapoda-research/llama-7b-hf')
    instruction = RECLETTER_PROMPTS[1].format(
        arguments["occupation"], arguments["name"], arguments["pronoun"]
    )
    utt = arguments["info"]
    utt = instruction + "\n" + utt
    try:
        input_ids = tokenizer.encode(utt)
        input_id_len = len(input_ids)
        input_ids = torch.tensor(input_ids, device=device, dtype=torch.long).unsqueeze(0)
        out = model.generate(
            input_ids,
            max_new_tokens=512,
            repetition_penalty=1.5,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=2,
        )[0]
        text = tokenizer.decode(
            out[input_id_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if text.find(tokenizer.eos_token) > 0:
            text = text[: text.find(tokenizer.eos_token)]
        # text = trim_text(text)
        text = text.strip()
    except Exception as e:
        print("Error: {}".format(e))
        text = ""
    print("Vicuna: {}".format(text))
    return text


# class StopOnTokens(StoppingCriteria):
#     def __call__(self, input_ids, scores, **kwargs):
#         stop_ids = [50278, 50279, 50277, 1, 0]
#         for stop_id in stop_ids:
#             if input_ids[0][-1] == stop_id:
#                 return True
#         return False


def generate_response_rec_stablelm(arguments, model, tokenizer, device):
    from transformers import StoppingCriteria, StoppingCriteriaList
    
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            stop_ids = [50278, 50279, 50277, 1, 0]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    utt = arguments["info"]
    system_prompt = RECLETTER_PROMPTS[1].format(
        arguments["occupation"], arguments["name"], arguments["pronoun"]
    )
    prompt = f"<|SYSTEM|>{system_prompt}<|USER|>{utt}<|ASSISTANT|>"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_id_len = inputs["input_ids"].size()[1]
        tokens = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
        )[0]
        text = tokenizer.decode(tokens[input_id_len:], skip_special_tokens=True)
        if text.find(tokenizer.eos_token) > 0:
            text = text[: text.find(tokenizer.eos_token)]
        # text = trim_text(text)
        text = text.strip()
    except Exception as e:
        print("Error: {}".format(e))
        text = ""
    print("StableLM: {}".format(text))
    return text
