#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

## To generate the 0-shot column of CoT prompting
tmux new-session -d -s llama3p1-8b-instruct
tmux send -t llama3p1-8b-instruct 'conda activate orqa_py3.11 && python cot_inference.py --model_name llama3p1-8b-instruct --n_shots 0' ENTER

tmux new-session -d -s llama3p1-70b-instruct
tmux send -t llama3p1-70b-instruct 'conda activate orqa_py3.11 && python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0' ENTER

tmux new-session -d -s llama3p1-405b-instruct
tmux send -t llama3p1-405b-instruct 'conda activate orqa_py3.11 && python cot_inference.py --model_name llama3p1-405b-instruct --n_shots 0' ENTER

tmux new-session -d -s llama3-8b-instruct
tmux send -t llama3-8b-instruct 'conda activate orqa_py3.11 && python cot_inference.py --model_name llama3-8b-instruct --n_shots 0' ENTER

tmux new-session -d -s llama3-70b-instruct
tmux send -t llama3-70b-instruct 'conda activate orqa_py3.11 && python cot_inference.py --model_name llama3-70b-instruct --n_shots 0' ENTER

tmux new-session -d -s llama2-7b-chat
tmux send -t llama2-7b-chat 'conda activate orqa_py3.11 && python cot_inference.py --model_name llama2-7b-chat --n_shots 0' ENTER

tmux new-session -d -s llama2-13b-chat
tmux send -t llama2-13b-chat 'conda activate orqa_py3.11 && python cot_inference.py --model_name llama2-13b-chat --n_shots 0' ENTER

tmux new-session -d -s llama2-70b-chat
tmux send -t llama2-70b-chat 'conda activate orqa_py3.11 && python cot_inference.py --model_name llama2-70b-chat --n_shots 0' ENTER

tmux new-session -d -s flan-t5-xxl
tmux send -t flan-t5-xxl 'conda activate orqa_py3.11 && python cot_inference.py --model_name flan-t5-xxl --n_shots 0' ENTER

tmux new-session -d -s falcon-7b-instruct
tmux send -t falcon-7b-instruct 'conda activate orqa_py3.11 && python cot_inference.py --model_name falcon-7b-instruct --n_shots 0' ENTER

tmux new-session -d -s deepseek-math-7b-instruct
tmux send -t deepseek-math-7b-instruct 'conda activate orqa_py3.11 && python cot_inference.py --model_name deepseek-math-7b-instruct --n_shots 0' ENTER

tmux new-session -d -s numinamath
tmux send -t numinamath 'conda activate orqa_py3.11 && python cot_inference.py --model_name numinamath --n_shots 0' ENTER

tmux new-session -d -s mistral_7b_instruct_v0dot1
tmux send -t mistral_7b_instruct_v0dot1 'conda activate orqa_py3.11 && python cot_inference.py --model_name mistral_7b_instruct_v0dot1 --n_shots 0' ENTER

tmux new-session -d -s mistral-7b-instruct-v0dot3
tmux send -t mistral-7b-instruct-v0dot3 'conda activate orqa_py3.11 && python cot_inference.py --model_name mistral-7b-instruct-v0dot3 --n_shots 0' ENTER

tmux new-session -d -s mixtral_8x7b
tmux send -t mixtral_8x7b 'conda activate orqa_py3.11 && python cot_inference.py --model_name mixtral_8x7b --n_shots 0' ENTER
