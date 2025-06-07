import json
import os

# If there is no trigger prompt set, use the default
if not os.getenv("trigger", ""):
    os.environ["trigger"] = "Let's think step by step"

import random
import time
from argparse import ArgumentParser
from typing import Dict, List

from final_inference_utils import HfInferenceRetriesClient, llm_hf_dict, ltr_to_idx
from tqdm import tqdm

import final_inference_utils

from task import StandardTask
import time

## This saves the LLM outputs into a folder called 'llm_predictions'
folder_path = final_inference_utils.get_folder_path("llm_predictions")

def get_config_and_api_key_name_from_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name", required=True, help='Model name (model names can be found in "src/final_inference_utils.py")'
    )
    parser.add_argument(
        "--n_shots", type=int, default=0, required=False, help='#/ In-context learning examples'
    )
    ## The huggingface API key may be added as a default value here
    parser.add_argument(
        "--token", type=str, default="hf_xxxx", required=False, help='Huggingface API key must be provided'
    )
    args = parser.parse_args()
    return args

def get_predictions_from_hf_inference(ds, model_name):
    cot_pipeline = HfInferenceRetriesClient(model=model_name, token=os.environ['API_TOKEN']).text_generation
    prediction = []
    counter = 0
    for ds_i in tqdm(ds):
        for _ in range(100):
            try:
                seq = cot_pipeline(
                    ds_i["input"], do_sample=False, max_new_tokens=1, return_full_text=False
                )
                break
            except:
                print("Error... retrying in a minute (this may be due to limit restrictions)")
                time.sleep(60)

        llm_result = -10
        llm_str_op = seq
        if llm_str_op.strip() in {"A", "B", "C", "D"}:
            llm_result = ltr_to_idx(llm_str_op)

        # If llm_result is not in the target_options, then it is incorrect
        if llm_result > 3 or llm_result < 0:
            llm_result = -1
            counter += 1

        prediction.append({"target": llm_result})

    print(f"LLM generated answer that was not in the target options: {counter} times")

    return prediction


def generate_llm_prediction(ds, type, n_shots, **kwargs) -> List[Dict]:
    """
    Return a list of dicts in the following format
    [{"target": answer1}, {"target": answer2}, ...]
    The list should be in the exact same order as the dataset (ds) passed.
    """

    if type in llm_hf_dict:
        prediction = get_predictions_from_hf_inference(ds, llm_hf_dict[type])
    elif os.path.isfile(f"{folder_path}/{type}.json"):
        with open(f"{folder_path}/{type}.json", "r") as f:
            predictions = json.load(f)
            return predictions
    else:
        raise Exception("ERROR: Model does not exist in the LLM registry")

    file_model_name = type.replace("/", "_")
    file_model_name = file_model_name.replace(".", "dot")
    os.makedirs(folder_path, exist_ok=True)
    with open(
        f"{folder_path}/{file_model_name}_{n_shots}shot_{str(time.time())}.json", "w+"
    ) as outfile:
        outfile.write(json.dumps(prediction, indent=4))

    return prediction


if __name__ == "__main__":
    config = get_config_and_api_key_name_from_args()
    
    token = config.token
    os.environ["API_TOKEN"] = token

    task = StandardTask(config.n_shots)
    ds = task.get_dataset()
            
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("Running model", config.model_name)
    
    llm_output = generate_llm_prediction(ds, config.model_name, config.n_shots)
    result = task.evaluate_predictions(predictions=llm_output, gold=ds)
    
    print("Score from experiment", result["accuracy"])
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
