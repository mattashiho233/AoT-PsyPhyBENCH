# AoT-PsyPhyBENCH: Evaluation Code and Data

This repository hosts the evaluation code and benchmark data for **AoT-PsyPhyBENCH**.  
It supports local **vLLM** inference, **OpenAI** API models, and **Google Gemini** models (via an OpenAI-compatible endpoint).

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