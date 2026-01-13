import argparse
from .util import make_normal_parser
from . import console
import readline

def args_parser():
    parser = make_normal_parser('fastllm_chat')
    args = parser.parse_args()
    return args

def fastllm_chat(args):
    from .util import make_normal_llm_model
    
    console.header("加载模型")
    model = make_normal_llm_model(args)

    generation_config = {
        'repetition_penalty': 1.0,
        'top_p': 0.8,
        'top_k': 1,
        'temperature': 1.0
    }
    import os
    import json
    if (os.path.exists(os.path.join(args.path, "generation_config.json"))):
        with open(os.path.join(args.path, "generation_config.json"), "r", encoding="utf-8") as file:
            config = json.load(file)
            if ('do_sample' in config and config['do_sample']):
                for it in ["repetition_penalty", "top_p", "top_k", "temperature"]:
                    if (it in config):
                        generation_config[it] = config[it];

    console.header("开始对话")
    console.info("输入 'clear' 清空记录, 'stop' 退出程序")
    history = []

    while True:
        query = input(console.user_prompt())
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            console.success("对话历史已清空")
            continue
        console.ai_response_start()
        curResponse = "";
        for response in model.stream_response(query, history = history, 
                                              repeat_penalty = generation_config["repetition_penalty"],
                                              top_p = generation_config["top_p"],
                                              top_k = generation_config["top_k"],
                                              temperature = generation_config["temperature"]):
            curResponse += response;
            print(response, flush = True, end = "")
        print()  # 换行
        history.append((query, curResponse))
    
    console.info("正在释放资源...")
    model.release_memory()
    console.success("已退出")

if __name__ == "__main__":
    args = args_parser()
    fastllm_chat(args)