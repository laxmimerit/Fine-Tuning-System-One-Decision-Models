<p align="center">
  <a href="https://kgptalkie.com/decision-ai-llm-fine-tuning">
    <img src="assets/banner.jpg" alt="Building and Fine-Tuning Decision AI LLM Transformers" width="100%">
  </a>
</p>

<h1 align="center">Fine-Tuning System One Decision Models</h1>

<p align="center">
  Code, data and cheat sheets for the KGP Talkie course<br>
  <b>Building and Fine-Tuning Decision AI LLM Transformers</b>
</p>

<p align="center">
  <a href="https://kgptalkie.com/decision-ai-llm-fine-tuning"><img src="https://img.shields.io/badge/Course-KGP%20Talkie-0D8A7D?style=for-the-badge" alt="Course page"></a>
  <a href="https://huggingface.co/kgptalkie/mini-laya"><img src="https://img.shields.io/badge/Model-kgptalkie%2Fmini--laya-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Mini Laya on the Hugging Face Hub"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/PyTorch-CUDA%2012.8%20%7C%20Apple%20MPS-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face Transformers">
  <img src="https://img.shields.io/badge/PEFT-LoRA%20%7C%20QLoRA-6E40C9" alt="PEFT">
  <img src="https://img.shields.io/badge/Ollama-GGUF-000000?logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white" alt="uv">
</p>

<p align="center">
  <a href="#what-you-build">What you build</a> ·
  <a href="#course-map">Course map</a> ·
  <a href="#repository-layout">Layout</a> ·
  <a href="#setup">Setup</a> ·
  <a href="#setup-on-a-mac-apple-silicon">Mac setup</a> ·
  <a href="#mini-laya-in-ollama-section-8">Mini Laya in Ollama</a> ·
  <a href="#known-issues">Known issues</a>
</p>

---

System One decision models such as Laya, Kev, tev1, nimble and clef-flash take a state plus typed questions (`choice`, `score`, `noul`) and return calibrated probabilities in a single forward pass, with no text generation. This course starts at the transformer underneath them, fine-tunes Laya on your own data, builds two decision models of your own (Mini Laya on an encoder, Mini Clef on a decoder), and ends by sharing Mini Laya on the Hugging Face Hub and running it in Ollama.

It follows **Jev and Laya Fundamentals**, which covers the Jev API, decision patterns and serving with Ollama.

## What you build

- Self-attention computed step by step, and the encoder and decoder families it powers
- Decision models run locally through Ollama's `/v1/systemone` endpoint, and the same kind of answer read by hand from an LLM's logits
- Laya fine-tuned on one problem statement (routing card messages) with the `laya` package
- **Mini Laya**: ModernBERT-base with Laya's decision head, trained with a proper scoring rule
- A classifier fine-tuned with the Hugging Face `Trainer`, then with LoRA and QLoRA through PEFT, including a 4B model in 4-bit
- **Mini Clef**: Qwen2.5-0.5B with a LoRA adapter and a decision head
- Mini Laya pushed to the Hub with a model card, loaded back by name, and converted to GGUF so Ollama runs it like Laya

## Course map

| Section | Topic | Code | Lab |
|---|---|---|---|
| 1 | Transformers: pipelines, tokens, attention, encoder and decoder, logits, generation | `01_P1`, `01_P2`, `01_P3` | Self-attention step by step |
| 2 | How a System One model decides: decision models on Ollama, answers read from an LLM's logits, calibration, Laya measured with `laya.evals` | `02` | Mistakes made with high confidence |
| 3 | Fine-tuning Laya on one problem statement with `laya`'s `finetune`: routing card messages | `03` | Original vs fine-tuned Laya on new messages |
| 4 | Build your own Mini Laya: ModernBERT plus Laya's decision head, trained with `finetune` and a proper scoring rule | `04` | Fine-tune Mini Laya on custom data |
| 5 | Fine-tuning fundamentals: a new head, freezing, memory per parameter, `Trainer`, learning rate, overfitting | `05` | Accuracy vs learning rate |
| 6 | LoRA, QLoRA and PEFT: adapters, save, load and merge, a 4B model in 4-bit | `06` | Accuracy vs LoRA rank |
| 7 | Build your own Mini Clef: Qwen2.5-0.5B with LoRA and a decision head, trained with `Trainer` | `07` | Laya vs Mini Laya vs Mini Clef on the card task |
| 8 | Share your model: Hub repositories, a model card, uploading from Python, and Mini Laya in Ollama | `ollama/` | Same answers from Python and from Ollama |

## Repository layout

```text
.
├── notebooks/            the code, one notebook per section (run them from this folder)
│   └── data/tickets.csv  the support tickets used in Sections 1 and 2
├── cheatsheets/          a two to five page PDF per notebook, same name as the notebook
├── ollama/               Section 8: write Mini Laya as GGUF and run it in Ollama
│   ├── mini_laya_to_gguf.py
│   ├── Modelfile
│   └── flight.json
├── assets/               the banner at the top of this page
├── pyproject.toml        dependencies (managed with uv)
└── uv.lock
```

Cheat sheets for Sections 5 to 7 are added as they are released.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- An NVIDIA GPU for Sections 3 to 7. The course was recorded on an RTX 5090 with 32 GB; the PyTorch wheels come from the CUDA 12.8 index, which RTX 50 series cards need.
- Or a Mac with Apple Silicon (M1 or newer), on this `macos` branch. See [Setup on a Mac](#setup-on-a-mac-apple-silicon).
- [Ollama](https://ollama.com) for Sections 2 and 8
- A free Hugging Face account and access token

## Setup

Install the dependencies:

```bash
uv sync
```

Sign in to Hugging Face once, so model downloads are not rate-limited. Without it the notebooks still run, but print a warning about unauthenticated requests.

```bash
uv run hf auth login
```

Section 2 runs three decision models through Ollama on port 11434. Pull them once:

```bash
ollama pull tev1
```

```bash
ollama pull nimble
```

```bash
ollama pull clef-flash
```

## Setup on a Mac (Apple Silicon)

This `macos` branch runs the course on a Mac with an M-series chip. PyTorch uses the Mac's own GPU through **MPS** (Metal Performance Shaders) instead of an NVIDIA GPU through CUDA. The course was recorded and tested on an NVIDIA GPU, so expect training to be slower on a Mac. The code is the same apart from the changes listed in [What changed for the Mac](#what-changed-for-the-mac).

### 1. Install the tools

You need Git, uv and Ollama. If you use [Homebrew](https://brew.sh):

```bash
brew install git uv
```

```bash
brew install --cask ollama
```

- **Git** downloads this repository and Kev's data.
- **uv** installs Python 3.12 and every library in `pyproject.toml` into a private `.venv` folder, so nothing touches the Python on your system.
- **Ollama** runs the decision models in Section 2 and your own Mini Laya in Section 8. It runs natively on Apple Silicon.

### 2. Get the `macos` branch

```bash
git clone -b macos https://github.com/laxmimerit/Fine-Tuning-System-One-Decision-Models.git
```

```bash
cd Fine-Tuning-System-One-Decision-Models
```

`-b macos` checks out this branch instead of `main`. The notebooks on `main` use `"cuda"` directly, so on a Mac they would quietly run on the CPU or fail.

### 3. Install the libraries

```bash
uv sync
```

On Windows and Linux, `pyproject.toml` takes PyTorch from the CUDA 12.8 index. On macOS it skips that index and installs the standard PyPI build of PyTorch, which includes MPS. `uv sync` picks the right one for your computer by itself.

### 4. Check that PyTorch sees the Mac GPU

```bash
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

It should print `True`. If it prints `False`, check that you are on an Apple Silicon Mac, not an Intel one, and that macOS is version 14 (Sonoma) or newer. The PyTorch build this course installs needs macOS 14.

### 5. Sign in to Hugging Face and pull the Ollama models

These are the same steps as in [Setup](#setup):

```bash
uv run hf auth login
```

```bash
ollama pull tev1
```

```bash
ollama pull nimble
```

```bash
ollama pull clef-flash
```

The Hugging Face login avoids download rate limits. The three Ollama models are the decision models Section 2 compares.

### 6. Get Kev's data and open the notebooks

```bash
git clone https://github.com/jaredpalmer/kev vendor/kev
```

```bash
uv run jupyter lab notebooks
```

Kev's labelled eval sets are used in Sections 3, 4 and 7 (see [Data for Sections 3 to 7](#data-for-sections-3-to-7)). Run the notebooks from the `notebooks` folder, in order.

### What changed for the Mac

Each notebook starts with one new cell that picks the device:

```python
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
```

On a Mac it prints `'mps'`. On an NVIDIA machine the same notebooks still pick `'cuda'`, so this branch works on both.

| On `main` | On `macos` | Why |
|---|---|---|
| `device_map="cuda"`, `.to("cuda")`, `.cuda()` | `device_map=DEVICE`, `.to(DEVICE)` | Moves models and tensors to whichever GPU you have |
| `laya.Agent(..., device="cuda")` | `laya.Agent(..., device=DEVICE)` | With `"cuda"` on a Mac, Laya falls back to the CPU with only a warning, which is much slower |
| `torch.cuda.reset_peak_memory_stats()`, `torch.cuda.max_memory_allocated()` | `reset_peak_memory()`, `gpu_memory()` | MPS has no peak-memory counter. On a Mac, `gpu_memory()` shows the memory Metal holds at that moment, so the numbers differ from the video |
| `bf16` / `fp16` mixed precision in `TrainingArguments` | On only when `DEVICE == "cuda"` | Mixed precision was only tested on CUDA; on a Mac the `Trainer` runs without it, which is slower but safe |
| QLoRA cells in Section 6 | Wrapped in `if QLORA:`, skipped on a Mac | `bitsandbytes` 4-bit loading needs an NVIDIA GPU |

The saved outputs in the notebooks still come from the recording on an NVIDIA GPU, so they show `device='cuda:0'`. Your outputs will show `mps:0`.

### What runs where

| Section | On a Mac |
|---|---|
| 1, 2, 3, 4, 5, 7 | Runs on MPS |
| 6 | LoRA, saving, loading, merging and the Lab run on MPS. The QLoRA part needs an NVIDIA GPU: run `06_LoRA_QLoRA_and_PEFT.ipynb` in Google Colab with a T4 GPU to try it |
| 8 | Runs as it is: `mini_laya_to_gguf.py` needs no GPU, and Ollama runs on Apple Silicon |

On the RTX 5090, training in Sections 5 and 6 peaked at about 8 GB of GPU memory. A Mac shares one pool of memory between the CPU and GPU, so close other large apps before you train. Without mixed precision, memory use on a Mac will be higher than in the video.

## Data for Sections 3 to 7

Kev's labelled eval sets come from its repository, cloned into `vendor/` next to `notebooks/`:

```bash
git clone https://github.com/jaredpalmer/kev vendor/kev
```

- The card problem uses Banking77 through its parquet mirror `mteb/banking77`. The original `PolyAI/banking77` uses a loading script that current `datasets` no longer runs.
- Section 3 tests what Laya forgot on Kev's transfer set, and Sections 4 and 7 train on Kev's eval sets in `vendor/kev/evals`. Held-out calibration, test and transfer rows are removed by text hash.
- Run the sections in order. Section 2 writes `notebooks/runs/tickets.jsonl`; Section 3 writes `notebooks/runs/cards` and `notebooks/runs/laya-cards`; Section 4 writes `notebooks/runs/system-one-data`, `notebooks/runs/mini-laya` and `notebooks/runs/mini-laya-cards`; Section 7 reads the data, writes `notebooks/runs/mini-clef`, and its lab compares all of them.

## Mini Laya in Ollama (Section 8)

`ollama/mini_laya_to_gguf.py` writes Mini Laya as one GGUF file in the layout of Ollama's own `laya` build (architecture `modern-bert`, decision type `laya`), so Ollama runs it as a decision model through `/v1/systemone`. Run these from the `ollama/` folder.

Download Mini Laya from the Hub, or point the converter at your own `notebooks/runs/mini-laya`:

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('kgptalkie/mini-laya', local_dir='from-hub/mini-laya')"
```

Write the GGUF file (about 314 MB, 16-bit weights):

```bash
uv run python mini_laya_to_gguf.py from-hub/mini-laya mini-laya.gguf
```

Create the model from the `Modelfile` next to it:

```bash
ollama create mini-laya
```

Ask it the flight questions:

```bash
curl http://localhost:11434/v1/systemone -d "@flight.json"
```

- On Windows PowerShell, type `curl.exe`, since `curl` there is a different command.
- Mini Clef does not take this route: Ollama's clef models use a different decision head, so Mini Clef runs in Python with its adapter.

## Known issues

- A notebook cell that displays a loss tensor keeps its autograd graph alive. On a long run this pushes GPU memory into shared system memory and training slows to a crawl. Display `loss.item()` instead.

---

## Author

Laxmi Kant Tiwari, [KGP Talkie](https://kgptalkie.com). Questions about the course go to the Q&A on the course page.
