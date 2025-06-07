from .base_task import Selection, Task

PROMPT_BUILDER = {
    "instruction_zero_shot": "Given the context (following Context:), select the most appropriate answer to the question (following Question:). Answer only 'A', 'B', 'C', or 'D'\n",
    "instruction_few_shot": "Given the context (following Context:), select the most appropriate answer to the question (following Question:). Answer only 'A', 'B', 'C', or 'D'\nHere are some examples: \n\n",
    "output_prefix": "\nAnswer: Among A through D, the answer is (",
    "few_shot_example_separator": ")\n\n",
}

class StandardTask(Task):
    def __init__(self, n_shots: int, selection_mode: Selection = Selection.RANDOM):
        super().__init__(n_shots, selection_mode)

    def get_dataset(self):
        dataset = []
        for datapoint in self.testdataset:
            new_datapoint = datapoint.copy()
            new_datapoint["input"] = self.build_input(datapoint, PROMPT_BUILDER)
            dataset.append(new_datapoint)

        return dataset
