import json
import os
import os

# If there is no trigger prompt set, use the default
if not os.getenv("trigger", ""):
    os.environ["trigger"] = "Let's think step by step"
    
import random
import time
import warnings
from argparse import ArgumentParser
from typing import Dict, List

from final_inference_utils import llm_hf_dict, ltr_to_idx
from tqdm import tqdm

import final_inference_utils

from task import COTTask

import random
from transformers import pipeline

retry_loop = 10

warnings.filterwarnings(
    "ignore", message="Unverified HTTPS request is being made to host*"
)

folder_path = final_inference_utils.get_folder_path("llm_predictions_cot")


def get_input_prompt_for_ans(reasoning_prompt, reasoning):
    return reasoning_prompt.format(reasoning=reasoning)

def get_config_and_api_key_name_from_args():
    parser = ArgumentParser()
    parser.add_argument("--ds_name", help="orqa", default="cot", required=False)
    parser.add_argument(
        "--model_name", required=True, help='Model name (model names can be found in "src/final_inference_utils.py")'
    )
    parser.add_argument(
        "--n_shots", type=int, default=0, required=False, help='#/ In-context learning examples'
    )
    parser.add_argument(
        "--token", type=str, default="hf_xxxx", required=False, help='Huggingface API key must be provided'
    )
    args = parser.parse_args()
    return args


def get_predictions_from_hf_inference(ds, ds_ans, model_name):
    pipe = pipeline("text-generation", model=model_name, use_auth_token=os.environ['API_TOKEN'])

    random.seed(None)

    prediction = []
    counter = 0

    for (
        ds_i,
        ds_ans_i,
    ) in tqdm(zip(ds, ds_ans), total=len(ds_ans)):
        for _ in range(retry_loop):
            try:
                seq = pipe(ds_i["input"], do_sample=False, max_new_tokens=500, return_full_text=False)[0]["generated_text"]
                break
            except:
                print("ERROR in reasoning... retrying in a minute (this may be due to limit restrictions)")
                time.sleep(60)

        reasoning_for_datapoint = seq
        new_input_prompt = get_input_prompt_for_ans(ds_ans_i["input"], seq)

        for _ in range(retry_loop):
            try:
                seq = pipe(new_input_prompt, do_sample=False, max_new_tokens=1, return_full_text=False)[0]["generated_text"]
                break
            except:
                print("ERROR in answering... retrying in a minute (this may be due to limit restrictions)")
                time.sleep(60)

        llm_result = -10
        llm_str_op = seq
        if llm_str_op.strip() in {"A", "B", "C", "D"}:
            llm_result = ltr_to_idx(llm_str_op)

        # If llm_result is not in the target_options then We are assuming the result to be wrong.
        if llm_result > 3 or llm_result < 0:
            llm_result = (ds_i["TARGET_ANSWER"] + 1) % len(ds_i["OPTIONS"])
            counter += 1

        prediction.append(
            {
                "reasoning_extraction_prompt": ds_i["input"],
                "reasoning": reasoning_for_datapoint,
                "answer_extraction_prompt": str(new_input_prompt),
                "actual_answer": seq,
                "target": llm_result,
            }
        )

    print(f"LLM generated answer that was not in the target options: {counter} times")

    return prediction


def generate_llm_prediction(ds, ds_ans, type_model, n_shots, **kwargs) -> List[Dict]:
    """
    Return a list of dicts in the following format
    [{"target": answer1}, {"target": answer2}, ...]
    The list should be in the exact same order as the dataset (ds) passed.
    """

    if type_model in llm_hf_dict:
        prediction = get_predictions_from_hf_inference(ds, ds_ans, llm_hf_dict[type_model])
    elif os.path.isfile(f"{folder_path}/{type_model}.json"):
        with open(f"{folder_path}/{type_model}.json", "r") as f:
            predictions = json.load(f)
            return predictions
    else:
        raise Exception("ERROR: Model does not exist in the LLM registry")

    file_model_name = type_model.replace("/", "_")
    file_model_name = file_model_name.replace(".", "dot")
    os.makedirs(folder_path, exist_ok=True)
    time_stamp = str(time.time())
    os.environ["save_path"] = f"{folder_path}/{file_model_name}_{n_shots}shot_{time_stamp}.json"
    with open(
        os.environ["save_path"], "w+"
    ) as outfile:
        outfile.write(json.dumps(prediction, indent=4))

    return prediction


if __name__ == "__main__":
    config = get_config_and_api_key_name_from_args()

    token = config.token
    os.environ["API_TOKEN"] = token

    task = COTTask(config.n_shots)

    ds = task.get_dataset(reasoning_extraction=True)

    ds_ans = task.get_dataset(reasoning_extraction=False)
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("Running model", config.model_name)
    llm_output = generate_llm_prediction(ds, ds_ans, config.model_name, config.n_shots)
    result = task.evaluate_predictions(predictions=llm_output, gold=ds_ans)

    print("Score from experiment", result["accuracy"])
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    timestr = time.strftime("%Y%m%d-%H%M%S")

    text_file = open(f"{timestr}-{config.model_name}-{config.n_shots}shots-output.txt", "w")

    text_file.write(f"Score from experiment: {result['accuracy']}\n\nTrigger prompt: {os.environ['trigger']}\n\nSave Path: {os.environ['save_path']}")

    text_file.close()
