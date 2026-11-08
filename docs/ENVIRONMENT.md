# Development environment: verified facts and implications

I inspected this machine on **2026-11-08**. Each item is marked:
- **[verified]**: I checked it on this machine with a command.
- **[coordinator]**: reported by the coordinator; I did not re-check it.
- **[research]**: taken from a public source; the source and date are given.

## 1. Machine

| Item | Value | How |
|---|---|---|
| OS | Windows 11 Home Single Language 10.0.26300 | [verified] env |
| CPU | Intel Core i5-11400H, 6 cores / 12 threads | [verified] `Win32_Processor` |
| RAM | 15.7 GiB (16,893,382,656 bytes) | [verified] `Win32_ComputerSystem` |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop, **4096 MiB VRAM**, WDDM. The driver reports CUDA UMD 13.3. Idle temperature was 66 °C in P3 at 14 W of a 30 W cap. | [verified] `nvidia-smi` |
| Free disk | C: 32.8 GB, X: 35.8 GB, E: 202.7 GB | [verified] `Get-PSDrive` |
| Git | 2.37.1.windows.1. Repo is on `main` with no commits. | [verified] |

## 2. Toolchain

| Tool | State | How |
|---|---|---|
| Python | **3.14.3 only** (`py -0p` → `C:\Python314\python.exe`). pip 25.3. | [verified] |
| Global packages | `torch 2.11.0+cpu` (CPU-only, `torch.cuda.is_available() == False`), torchvision 0.26, numpy 2.4.4, uvicorn 0.46 | [verified] `pip list` and a torch probe |
| uv | not installed | [verified] |
| Node / npm | Node v24.13.0, npm 11.6.2. pnpm is not installed. | [verified] |
| Docker | CLI 29.8.1, context `desktop-linux`. WSL default distro is `docker-desktop`. **Daemon not running**: `failed to connect to ... dockerDesktopLinuxEngine`. | [verified] |
| Ollama | not installed | [verified] |
| PostgreSQL / Qdrant | no local install (`psql` not found). They will run in Docker. | [verified] |

## 3. Implications

### 3.1 Python version: pin 3.13 via uv and leave the system 3.14 alone
- **ragas 0.4.3 needs `scikit-network`, which blocks 3.14.** ragas 0.4.3 (2026-01-13) has a hard dependency on `scikit-network` [verified via the PyPI JSON on 2026-11-08]. scikit-network 0.33.5 ships `win_amd64` wheels for **cp310–cp313 only**, with no cp314 [verified via PyPI JSON]. Without a wheel, 3.14 means building C++ from source with MSVC.
- **langgraph is only tagged to 3.13.** langgraph 1.2.20 declares classifiers up to 3.13. It is pure Python, so it would probably run on 3.14, but nobody has tested it there [research: pypi.org/pypi/langgraph, 2026-11-08].
- **PyTorch has CUDA wheels for 3.13.** torch 2.15.0 publishes `+cu132` Windows wheels for cp311–cp315 [verified: listing at download.pytorch.org/whl/cu132/torch/].
  - Ampere sm_86 is still supported, and CUDA 13 wheels need driver ≥ 580 [research: PyTorch 2.15 release notes and the dev-discuss notice of 2026-09-04].
  - The driver reports CUDA 13.3, so cu132 should work. I have not run it: check with `torch.cuda.get_arch_list()` after installing.
- **Decision: Python 3.13**, installed by uv into a project `.venv` (see ADR-0002). Why not 3.12: it gets security fixes only, and 3.13 has every wheel we need.
- **The global CPU-only torch must not be used.** The project venv installs its own CUDA build.
- **Setup steps for the user (not done yet; they need approval):**
  1. `py -m pip install --user uv`, as the coordinator suggested. The standalone installer (`irm https://astral.sh/uv/install.ps1 | iex`) is the only method that supports `uv self update` [research: docs.astral.sh/uv, 2026-11-08]. Either one works.
  2. `uv python install 3.13`, then `uv python pin 3.13` (writes `.python-version`).
  3. Install nothing heavy globally.

### 3.2 GPU (4 GB VRAM): what fits

The numbers below are **estimates, not measurements**. They use parameter counts from the model cards and these rules of thumb:
- fp16 inference: about 2 bytes per parameter for weights, plus activations.
- Full fine-tuning with AdamW and mixed precision: about 16 bytes per parameter (fp32 master weights, gradients and two Adam moments), plus activations.

WDDM and the desktop compositor also take part of the 4 GB, so plan on about 3.3–3.5 GB being usable.

| Workload | Model (params) | Estimate | Verdict |
|---|---|---|---|
| Embedding, inference | bge-small-en-v1.5 (33M), e5-small-v2 (33M), all-MiniLM-L6-v2 (23M) | Weights under 100 MB in fp16. Batch 64–128 at 512 tokens fits. | ✅ GPU, and fine on CPU too |
| Embedding, inference | bge-base-en-v1.5 (~110M) | About 220 MB of weights | ✅ GPU. CPU is acceptable for ≤ 20k chunks. |
| Embedding, inference | Qwen3-Embedding-0.6B | About 1.2 GB of weights in fp16. Keep the batch small and seq ≤ 512. | ⚠ GPU only, experiments only |
| Cross-encoder, inference | ms-marco-MiniLM-L6-v2 (23M) | Trivial | ✅ GPU or CPU |
| Cross-encoder, inference | bge-reranker-base (278M), gte-reranker-modernbert-base (149M) | 0.3–0.6 GB of weights | ✅ GPU. Slow on CPU at rerank depth 50. |
| Reranker **fine-tune** | MiniLM-L6/L12 (23–33M) | About 0.5 GB of optimizer state. Batch 32 at seq 256 fits. | ✅ Primary fine-tuning target |
| Reranker **fine-tune** | gte-reranker-modernbert-base (149M) | About 2.4 GB of state. Batch 4–8 at seq 256, with gradient checkpointing and bf16/fp16 AMP. | ⚠ Feasible but tight. Secondary target. |
| Reranker **fine-tune** | bge-reranker-base (278M; ~192M of it is the 250k-vocab embedding matrix) | Full fine-tuning needs about 4.5 GB, which does not fit. Freezing the embedding layer leaves about 86M trainable parameters. | ⚠ Only with frozen embeddings |
| Reranker **fine-tune** | bge-reranker-v2-m3, Qwen3-Reranker-0.6B | About 9 GB or more | ❌ Unless we use LoRA, which is not verified with `CrossEncoderTrainer` |

Other GPU constraints:
- **Thermals.** The laptop GPU idled at 66 °C. Expect throttling on long runs, so fine-tuning runs must checkpoint and resume, and record their wall-clock time.
- **CPU fallback.** Every model path must run on CPU with `device="cpu"`; the GPU is an accelerator, never a requirement.
  - On CPU (EST), indexing 5–20k chunks takes minutes with a ~33M encoder and tens of minutes with a ~110M one.
  - On CPU, reranking 50 candidates with MiniLM-L6 should be sub-second to about 1 s (EST). bge-reranker-base would take several seconds (EST).
  - So the CPU default reranker is MiniLM-L6.
- **GPU inside Docker.** Docker Desktop documents GPU passthrough through WSL2, but I have not verified it here. Training and experiment jobs therefore run **natively on Windows through uv**, where CUDA is available. The containerized API runs embedding and reranking on CPU by default, with an optional GPU compose profile.

### 3.3 Docker
- **The user must start Docker Desktop** before anything in M1 that touches Postgres or Qdrant can be tested.
- **Disk.** Docker Desktop's WSL disk lives on C: by default, which has 32.8 GB free. Planned images:

  | Image | Size |
  |---|---|
  | python:3.13-slim + torch CPU | about 1.5–2.5 GB (EST) |
  | postgres | about 0.4 GB |
  | qdrant | about 0.2 GB |
  | mlflow | about 1 GB |
  | node | about 0.3 GB |

  That fits, but consider moving Docker's disk image to E:.
- **Memory budget (EST):**

  | Component | RAM |
  |---|---|
  | Docker VM overhead | about 1.5 GB |
  | Postgres | 0.3–0.5 GB |
  | Qdrant with ≤ 50k 384-dim points | under 0.5 GB |
  | API with embedder and reranker loaded | 1.5–2.5 GB |
  | Next.js dev server | about 1 GB |
  | MLflow | about 0.3 GB |
  | **Total** | **about 5–6 GB, which fits in 15.7 GB** |

  **A self-hosted Langfuse does not fit comfortably.** Its docs recommend ≥ 4 cores and 16 GiB for the compose stack (web, worker, Postgres, ClickHouse, Redis, MinIO) [research: langfuse.com/self-hosting/deployment/docker-compose, 2026-11-08]. See ADR-0014.

### 3.4 Model and dataset caches
- Hugging Face models plus BEIR data come to about 2–5 GB. Set `HF_HOME` to a folder on E: or X: (for example `X:\projects\mezvuat-ai\.cache\hf`, which is git-ignored) to keep C: free.
- Before downloading models or datasets, ask the user and state the name, source and size.

### 3.5 Frontend
- Node 24 is fine: Next.js 16.4.x requires Node ≥ 20.9 [research: registry.npmjs.org/next, 2026-11-08].
- **npm** is used, so pnpm and corepack are not needed. Corepack is no longer bundled from Node 25 on [research, secondary source].

### 3.6 Services that are not available
- **No LLM API keys and no Ollama.** Every LLM-dependent milestone (M4 onward) is blocked until the user provides a direct-provider key (ADR-0011: Groq) or a ModelMesh endpoint and key.
- **Without any key, these still run:** ingestion, retrieval, the evaluation harness and reranker training.
- Secrets live only in `.env`, which is git-ignored. `.env.example` lists the variable names.

## 4. Things I could not verify
- That CUDA actually works with torch 2.15+cu132 on this driver. Nothing has been installed yet.
- GPU passthrough into Docker Desktop on this machine.
- Real throughput and latency for any model here. M2 and M3 will measure them; the numbers above are estimates.
