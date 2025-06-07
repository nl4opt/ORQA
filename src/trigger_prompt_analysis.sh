#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

conda activate orqa_py3.11

export trigger="Let's think step by step"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's work by elimination"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's reflect on each answer option like an operations research expert"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's use step by step inductive reasoning, given the mathematical nature of the question"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's think step by step like an operations research expert"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's think step by step"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's work by elimination"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's reflect on each answer option like an operations research expert"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's use step by step inductive reasoning, given the mathematical nature of the question"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's think step by step like an operations research expert"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's think step by step"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's work by elimination"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's reflect on each answer option like an operations research expert"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's use step by step inductive reasoning, given the mathematical nature of the question"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's think step by step like an operations research expert"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's think step by step"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's work by elimination"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's reflect on each answer option like an operations research expert"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's use step by step inductive reasoning, given the mathematical nature of the question"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's think step by step like an operations research expert"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's think step by step"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's work by elimination"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's reflect on each answer option like an operations research expert"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's use step by step inductive reasoning, given the mathematical nature of the question"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait

export trigger="Let's think step by step like an operations research expert"
python cot_inference.py --model_name llama3p1-70b-instruct --n_shots 0 --do_sample 1
wait
