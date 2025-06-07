import random
from typing import Dict, Any
import evaluate
from datasets import load_dataset
from enum import Enum
from final_inference_utils import get_folder_task_path

random.seed(100)
# get_folder_task_path may not work in this folder. Hence when running the code just cross check the relative paths once.
# Should be easy enough to fix if any error comes up so I am not testing it for now.
DATA_SOURCE = {
    "test": get_folder_task_path("dataset/ORQA_test.jsonl"),  # link to our sample test set
    "validation": get_folder_task_path("dataset/ORQA_validation.jsonl"),  # link to our sample dev set
}

def idx_to_ltr(idx):
    return chr(int(idx) + ord("A"))


def ltr_to_idx(ltr):
    return ord(ltr) - ord("A")


class Selection(Enum):
    RANDOM = 1
    SAME_TYPE = 2


class Task:
    def __init__(self, n_shots: int, selection_mode: Selection = Selection.RANDOM):
        self.n_shots = n_shots
        self.dataset = load_dataset("json", data_files=DATA_SOURCE)
        self.testdataset = self.dataset["test"]
        self.valdataset = self.dataset["validation"]
        self.selection_mode = Selection.SAME_TYPE # fixing this for now

    def get_val_points(self, datapoint):
        if self.selection_mode == Selection.SAME_TYPE:
            return self.get_same_type_val_points(datapoint)
        else:
            return self.get_random_val_points()

    def get_random_val_points(self):
        return random.choices(self.valdataset, k=self.n_shots)

    def get_same_type_val_points(self, datapoint):
        val_dataset = []
        instances_len = []

        ## random question type
        for val_point in self.valdataset:
            val_dataset.append(val_point)
            instances_len.append(len(val_point["CONTEXT"]))

        ## Random length
        return random.choices(val_dataset, k=self.n_shots)

    def get_input_string(self, example: Dict[str, Any]):
        context_NL = example["CONTEXT"]
        question_NL = example["QUESTION"]
        option_0 = example["OPTIONS"][0]
        option_1 = example["OPTIONS"][1]
        option_2 = example["OPTIONS"][2]
        option_3 = example["OPTIONS"][3]

        input_str = (
            "Context: "
            + str(context_NL)
            + "\nQuestion: "
            + str(question_NL)
            + "\nA. "
            + str(option_0)
            + "\nB. "
            + str(option_1)
            + "\nC. "
            + str(option_2)
            + "\nD. "
            + str(option_3)
        )

        return input_str

    def get_target_string(self, val_point):
        return idx_to_ltr(val_point["TARGET_ANSWER"])

    def build_input(self, datapoint, prompt_builder):
        input_string = self.get_input_string(datapoint)

        if self.n_shots == 0:
            prefix = prompt_builder["instruction_zero_shot"]
        else:
            prefix = prompt_builder["instruction_few_shot"]
            val_points = self.get_val_points(datapoint)

            for val_point in val_points:
                prefix = (
                    prefix
                    + self.get_input_string(val_point)
                    + prompt_builder["output_prefix"]
                )
                prefix = (
                    prefix
                    + self.get_target_string(val_point)
                    + prompt_builder["few_shot_example_separator"]
                )

        return prefix + input_string + prompt_builder["output_prefix"]

    def evaluate_predictions(self, predictions, gold):
        metric = evaluate.load("accuracy")
        preds = [p["target"] for p in predictions]
        targets = [g["TARGET_ANSWER"] for g in gold]
        return metric.compute(references=targets, predictions=preds)
