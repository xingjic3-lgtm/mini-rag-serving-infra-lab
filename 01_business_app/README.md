# Phase 01: Business App

This phase builds the smallest local document QA business loop:

```text
fixed document -> fixed question -> retrieve context -> build prompt -> local LLM answer
```

## Model

The local model is:

```text
models/Qwen2.5-1.5B-Instruct
```

It is small enough for the RTX 5060 Ti setup and gives a better QA baseline than a 0.5B model.

## Run

Use the CUDA learning environment:

```powershell
conda activate cuda-lab
python 01_business_app/app.py
```

Or run it directly with the environment Python:

```powershell
D:\anaconda3\envs\cuda-lab\python.exe 01_business_app\app.py
```

Or use the project run script:

```powershell
.\run_phase01.ps1
```

To run the browser business app:

```powershell
.\run_phase01_web.ps1
```

Then open:

```text
http://127.0.0.1:8001
```

## Expected Output

The program should print:

```text
[Document Loaded]
[Question]
[Retrieved Context]
[Prompt]
[LLM Answer]
```

This completes the Phase 01 business loop. Phase 02 can benchmark the same request path.
