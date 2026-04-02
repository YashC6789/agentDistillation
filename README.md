# agentDistillation

Generated around 5.1k trajectories using generate_hf_trajectories_v100.py. When we get access to a better GPU we can re-generate these with a better model as well. 

For SFT pipeline, you can first use the gsm8k_golden_trajectories.jsonl for testing (around 300 trajectories), then you can run it on the full trajectory generation (gsm8k_golden_trajectories_new.jsonl). 

pip installs needed for trajectory generation: vllm datasets autoawq, i used python 3.11 conda env
