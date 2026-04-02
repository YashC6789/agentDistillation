# agentDistillation

Generated around 5.1k trajectories using generate_hf_trajectories_v100.py. When we get access to a better GPU we can re-generate these with a better model as well. 

For SFT pipeline, you can first use the gsm8k_golden_trajectories.jsonl for testing (around 300 trajectories), then you can run it on the full trajectory generation (gsm8k_golden_trajectories_new.jsonl). 

pip installs needed for trajectory generation: vllm datasets autoawq, i used python 3.11 conda env

pip installs needed for sft: 
- pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
- pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
- optional (pip uninstall autoawq -y), or you can force it to not use autoawq

To run the train_sft and eval_sft scripts, you need the pip installs above^. Also, change the file directories in these two files based off where you stored them. 