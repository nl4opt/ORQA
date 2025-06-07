from task import COTTask
import json
import os

## Point to the path with the text files
## This is used to extract which Trigger prompt was used
text_path = "ai4model_orqa/src"

## Point this path to the name of the output folder 
## containing the files to evaluate and extract accuracy & f1
path = "ai4model_orqa/src/output_of_llms/llm_predictions"

map_file_to_trigger = dict()

from pathlib import Path
for file in Path(text_path).rglob('*.txt'):
    txtFile = file.read_text()

    if "Save Path: " in txtFile:
        text_filename = txtFile.split("src/output_of_llms/llm_predictions/")[-1]
        trigger_prompt = txtFile.split("Trigger prompt: ")[-1].split("\n")[0]
        map_file_to_trigger[text_filename] = trigger_prompt

filelist = os.listdir(path)
outstr = ""
for file_name in filelist:
    os.environ["trigger"] = map_file_to_trigger[file_name]
    n_shots = int(file_name.split('shot')[0].split('_')[-1])
    if n_shots > 0:
        n_shots = 1
    else:
        n_shots = 0
    task = COTTask(int(n_shots))

    mapping = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
    }

    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score

    ds = task.get_dataset(reasoning_extraction=True)
    ds_ans = task.get_dataset(reasoning_extraction=False)

    with open(
        os.path.join(path, file_name),
    ) as jf:
        json_list = json.load(jf)

    pred = []
    gt = []
    skipping = 0
    for j in json_list:
        for d, d_ans in zip(ds, ds_ans):
            input_text_str = d["input"]
            input_context = input_text_str.split('Context:')[-1].split("A. ")[0] # contains question and also the context

            if input_context in j["reasoning_extraction_prompt"]:                
                j["actual_answer"] = j["actual_answer"][0]
                if "actual_answer" in j:
                    if j["actual_answer"] in mapping:
                        pred.append(mapping[j["actual_answer"]])
                        gt.append(d["target"])
                    else:
                        skipping += 1
                        # force an incorrect
                        pred.append(0)
                        gt.append(3)

    f1 = f1_score(gt, pred, average='macro')
    acc = accuracy_score(gt, pred)

    print(f"trigger: {os.environ["trigger"]}    acc: {acc}   f1: {f1}   skipped: {skipping}\n")
    outstr += f"trigger: {os.environ["trigger"]}    acc: {acc}   f1: {f1}   skipped: {skipping}\n"

import time
timestr = time.strftime("%Y%m%d-%H%M%S")
text_file = open(f"{timestr}-eval_trigger.txt", "w")
text_file.write(outstr)
text_file.close()