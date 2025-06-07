from .base_task import Selection, Task
from typing import Dict, Any
import random
import os

random.seed(100)

REASONING_PROMPT_BUILDER = {
    "instruction_zero_shot": "Given the context (following Context:), provide the chain of thoughts (following Reasoning:) to solve the question (following Question:). Remember, only one option is correct.\n",
    "instruction_few_shot": "Given the context (following Context:), provide the chain of thoughts (following Reasoning:) to solve the question (following Question:). Remember, only one option is correct.\n\nHere are some examples: \n\n",
    "output_prefix": f"\nReasoning: {os.environ['trigger']} ",
    "few_shot_example_separator": "\n\n",
}

ANSWER_PROMPT_BUILDER = {
    "instruction_zero_shot": "Given the context (following Context:), the reasoning (following Reasoning:), select the most appropriate answer to the question (following Question:). Answer only 'A', 'B', 'C', or 'D'. There is only one correct answer.\n",
    "instruction_few_shot": "Given the context (following Context:), the reasoning (following Reasoning:), select the most appropriate answer to the question (following Question:). Answer only 'A', 'B', 'C', or 'D'. There is only one correct answer.\n\nHere are some examples: \n\n",
    "output_prefix": "\nAnswer: Among A through D, the answer is (",
    "few_shot_example_separator": ")\n\n",
}


class COTTask(Task):
    def __init__(self, n_shots: int, selection_mode: Selection = Selection.SAME_TYPE):
        super().__init__(n_shots, selection_mode)
        self.reasoning_extraction = False

    def get_dataset(self, reasoning_extraction=False):
        dataset = []
        self.reasoning_extraction = reasoning_extraction
        for datapoint in self.testdataset:
            new_datapoint = datapoint.copy()
            if reasoning_extraction:
                new_datapoint["input"] = self.build_input(
                    datapoint, REASONING_PROMPT_BUILDER
                )
            else:
                new_datapoint["input"] = self.build_input(
                    datapoint, ANSWER_PROMPT_BUILDER
                )
            dataset.append(new_datapoint)

        return dataset

    def get_input_string(self, example: Dict[str, Any]):
        input_str = super().get_input_string(example)

        reasoning = example["REASONING"]

        if not self.reasoning_extraction:
            input_str = (
                input_str + f"\nReasoning: {os.environ['trigger']}. " + reasoning
            )

        return input_str

    def get_target_string(self, val_point):
        if self.reasoning_extraction:
            return val_point["REASONING"]
        return super().get_target_string(val_point)
