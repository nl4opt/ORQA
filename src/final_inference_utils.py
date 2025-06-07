import sys
from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt
import os
from dataclasses import dataclass

RETRIES = 100

# LLM Parameters
model_7b_chat = "meta-llama/Llama-2-7b-chat-hf"
model_70b_chat = "meta-llama/Llama-2-70b-chat-hf"
model_13b_chat = "meta-llama/Llama-2-13b-chat-hf"

llama3_8b_instruct = "meta-llama/Meta-Llama-3-8B-Instruct"
llama3_70b_instruct = "meta-llama/Meta-Llama-3-70B-Instruct"

llama3p1_8b_instruct = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llama3p1_70b_instruct = "meta-llama/Meta-Llama-3.1-70B-Instruct"
llama3p1_405b_instruct = "meta-llama/Meta-Llama-3.1-405B-Instruct"

flan_t5_11b = "google/flan-t5-xxl"
falcon_7b_instruct = "tiiuae/falcon-7b-instruct"

mixtral_8x7b = "mistralai/Mixtral-8x7B-Instruct-v0.1"
mistral_7b_instruct_v0dot1 = "mistralai/Mistral-7B-Instruct-v0.1"
mistral_7b_instruct_v0dot3 = "mistralai/Mistral-7B-Instruct-v0.3"

deepseek_instruct = "deepseek-ai/deepseek-math-7b-instruct"
numinamath = "AI-MO/NuminaMath-7B-TIR"

llm_hf_dict = {
    "mixtral_8x7b": mixtral_8x7b,
    "numinamath": numinamath,
    "llama2-7b-chat": model_7b_chat,
    "llama2-13b-chat": model_13b_chat,
    "llama2-70b-chat": model_70b_chat,
    "llama3-8b-instruct": llama3_8b_instruct,
    "llama3-70b-instruct": llama3_70b_instruct,
    "llama3p1-8b-instruct": llama3p1_8b_instruct,
    "llama3p1-70b-instruct": llama3p1_70b_instruct,
    "llama3p1-405b-instruct": llama3p1_405b_instruct,
    "deepseek-math-7b-instruct": deepseek_instruct,
    "flan-t5-xxl": flan_t5_11b,
    "falcon-7b-instruct": falcon_7b_instruct,
    "mistral_7b_instruct_v0dot1": mistral_7b_instruct_v0dot1,
    "mistral-7b-instruct-v0dot3": mistral_7b_instruct_v0dot3,
}
###########################

def ltr_to_idx(ltr):
    if len(ltr) != 1:
        return -10
    return ord(ltr) - ord("A")

def get_folder_path(experiment_name):
    folder_path = sys.path[0] + "/output_of_llms/" + experiment_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def get_folder_task_path(file_with_folder_name):
    folder_path = sys.path[0] + "/task/" + file_with_folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


class HfInferenceRetriesClient(InferenceClient):
    """
    Utility class to make multiple retries before failing.
    We use the tenacity library to achieve this.
    Currently the retries count is hardcoded. But could be extended to make it an argument to the class.
    """

    @retry(stop=stop_after_attempt(RETRIES))
    def chat_completion(self, prompt, *args, **kwargs):
        return super().chat_completion(prompt, *args, **kwargs)

class Exp:
    STANDARD = "standard_inference.py"
    COT = "cot_inference.py"

@dataclass
class Config:
    n_shot: int
    exp_type: str

if __name__ == "__main__":
    print("\n".join(llm_hf_dict.keys()))

    config = Config(0, Exp.STANDARD)

    with open(get_folder_path("tmux_script") + "/standard_0shot.sh", "w") as f:
        f.write("#!/bin/bash\n")

        for k in llm_hf_dict.keys():
            f.write(f"tmux new-session -d -s {k}_{config.n_shot}shot\n")
            f.write(
                f"tmux send -t {k}_{config.n_shot}shot 'conda activate mydevhuggingface && python {config.exp_type} --model_name {k} --n_shots {config.n_shot} --verbose 0' ENTER\n"
            )
