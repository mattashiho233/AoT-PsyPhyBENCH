# 🧠 🔄 AoT-PsyPhyBENCH: A psychophysically validated benchmark testing whether vision-language models can infer the arrow of time like humans do.

> **⚠️ Note: This repository is currently under construction. Content and structure may change.**

> 🏆 **Leaderboard coming soon!**

This repository hosts the evaluation code and benchmark data for **AoT-PsyPhyBENCH**: a psychophysically validated benchmark that tests whether VLMs can infer temporal direction in natural videos using the same stimuli and behavioral baselines established for humans.  
It supports local **vLLM** inference, **OpenAI** API models, and **Google Gemini** models (via an OpenAI-compatible endpoint).

## What is the Arrow-of-Time (AoT) task?
The Arrow-of-Time (AoT) task requires a vision-language model (VLM) to determine the playback direction of a video clip (a binary classification problem).
There is a substantial performance gap between humans and VLMs: even the latest proprietary VLM achieves only around 60% accuracy, whereas humans reach 90.1% on our benchmark.
![aot](overview.jpg)

## AoT-PsyPhyBENCH dataset download
Please refer to [`neuro_paper_data/neuro_paper_data/README.md`](./neuro_paper_data/neuro_paper_data/README.md)

## Leaderboard

The table below summarizes current results on AoT-PsyPhyBENCH. All models are evaluated using the official scripts in this repository.
## Zero-shot performance on AoT-PsyPhyBENCH

| Family        | Model               | Reasoning?      | F. F1 | B. F1 | Acc. |
|---------------|---------------------|-----------------|------:|------:|-----:|
| Baselines     | Random              | —               |   —   |   —   | 50.0 |
| Baselines     | Human               | —               | 90.0  | 88.0  | **89.2** |
| Open Models   | Qwen2-VL-7B         | Non-reasoning   | 66.7  |  0.0  | 50.0 |
| Open Models   | Qwen2.5VL-7B        | Non-reasoning   | 63.0  | 19.5  | 49.3 |
| Open Models   | Qwen2.5VL-72B       | Non-reasoning   | 57.4  | 38.2  | 49.5 |
| Open Models   | QVQ-72B-Preview     | Reasoning       | 66.1  |  0.0  | 49.4 |
| Open Models   | cosmos-reason1 7B   | Reasoning       | 31.2  | 63.3  | **52.1** |
| Proprietary   | GPT-4o              | Non-reasoning   | 65.4  | 24.9  | 52.6 |
| Proprietary   | GPT-4.1             | Non-reasoning   | 62.5  | 57.4  | **60.1** |
| Proprietary   | o3                  | Reasoning       | 67.2  | 29.1  | 55.2 |
| Proprietary   | o4-mini             | Reasoning       | 67.4  | 33.1  | 56.1 |
| Proprietary   | GPT-5               | Reasoning       | 68.7  | 26.8  | 56.1 |
| Proprietary   | Gemini-2.5-pro      | Reasoning       | 65.9  | 51.4  | 59.9 |




## Environment
Use **uv** to reconstruct the environment:
```bash
uv sync
```

## Run the Evaluation
The entrypoint scripts are `run_*.sh`. 

1. Replace the API key with yours in the entrypoint script. Adjust the concurrent call number according to your API rate limits. 
2. Check if the video data is correctly placed in the path as written in the entrypoint script. If you haven't prepared the video data, please follow the README in `./neuro_paper_data/neuro_paper_data/README`. 
3. Run the script. 

## 📄 Citation

If you use this benchmark in your research, please cite:
```bibtex
@article{matta2025whichway,
  title={Which Way Does Time Flow? A Psychophysics-Grounded Evaluation for Vision-Language Models},
  author={Matta, Shiho and Kanashiro Pereira, Lis and Han, Peitao and Cheng, Fei and Kitazawa, Shigeru},
  journal={arXiv preprint arXiv:2510.26241},
  year={2025},
  url={https://arxiv.org/abs/2510.26241}
}
```

**Paper:** [Which Way Does Time Flow? A Psychophysics-Grounded Evaluation for Vision-Language Models](https://arxiv.org/abs/2510.26241)
