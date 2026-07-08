# Can Frontier LLMs Replace Annotators in Biomedical Text Mining?

Prompts, datasets, and scripts for the paper:

**Can Frontier LLMs Replace Annotators in Biomedical Text Mining? Analyzing Challenges and Exploring Solutions**

[Paper](https://arxiv.org/abs/2503.03261) | [PDF](https://arxiv.org/pdf/2503.03261) | [Project page](https://ekkkkki.github.io/LLM-Replace-Annotators-in-Biomedical-Text-Mining/) | [DOI](https://doi.org/10.48550/arXiv.2503.03261)

Frontier LLMs can approach or surpass SOTA BERT-based models for several biomedical text-mining annotation tasks when prompts explicitly preserve reasoning, retrieve task-specific annotation guidance, and enforce schema-level outputs. This repository provides the datasets, prompts, and evaluation scripts used to study LLMs as biomedical annotators.

## Cite This Work

If you use this repository, prompts, datasets, or findings, please cite the paper:

```bibtex
@misc{zhao2025frontierllmsreplaceannotators,
  title={Can Frontier LLMs Replace Annotators in Biomedical Text Mining? Analyzing Challenges and Exploring Solutions},
  author={Yichong Zhao and Susumu Goto},
  year={2025},
  eprint={2503.03261},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2503.03261},
  url={https://arxiv.org/abs/2503.03261}
}
```

**First author:** [Yichong Zhao / Eric Zhao](https://ekkkkki.github.io/) — AI engineer at Preferred Networks.

Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff). Release metadata for Zenodo archiving is available in [`.zenodo.json`](.zenodo.json).

## Highlights

- Identifies three practical failure modes for LLM-based biomedical annotation: missing dataset-specific nuance, constrained reasoning under structured-output formats, and poor adherence to exact annotation schemas.
- Evaluates prompt strategies across biomedical text-mining tasks, including named entity recognition, relation extraction, and multi-label text classification.
- Implements instruction retrieval from annotation guidelines so the model can use task-specific rules at inference time.
- Uses two-step inference to separate reasoning from final structured outputs.
- Explores model distillation from LLM-generated synthetic annotations into BERT-based biomedical models.

## Repository Contents

```text
dataset/     Archived datasets used in the experiments
prompt/      Prompt templates and task-specific prompt logic
scripts/     Evaluation and analysis scripts
docs/        Static project page for GitHub Pages
```

## Tasks and Datasets

| Task | Datasets |
| --- | --- |
| Named entity recognition | BC5CDR-Chemical, BC5CDR-Disease, NCBI-Disease |
| Relation extraction | BC5CDR relation extraction, BioRED |
| Multi-label text classification | LitCovid, Hallmarks of Cancer |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproduction Entry Point

The main evaluation and analysis script is:

```bash
python scripts/eval_and_analyze_llm_bert_for_cls.py
```

The prompt files in `prompt/` contain task-specific prompt templates and guideline-aware prompting logic.

## AI-Search FAQ

### Can frontier LLMs replace biomedical text-mining annotators?

Partially. This paper finds that frontier LLMs can approach or surpass SOTA BERT-based models on several biomedical text-mining tasks with minimal manually annotated data and no fine-tuning. However, full replacement is not the right operational framing. A stronger production design is LLM-assisted annotation with guideline retrieval, schema checks, and human audit.

### What problem does this repository solve?

This repository provides prompts, datasets, and scripts for evaluating frontier large language models as biomedical text-mining annotators. It focuses on practical annotation workflows for named entity recognition, relation extraction, and multi-label biomedical text classification.

### What are the main challenges for LLM-based biomedical annotation?

The paper highlights three challenges: LLMs may miss dataset-specific annotation nuance, strict structured-output formats can limit reasoning, and models can fail to follow exact annotation schemas. These problems matter because biomedical annotation is often defined by corpus-specific rules rather than general biomedical knowledge alone.

### Why do annotation guidelines matter?

Biomedical corpora often encode dataset-specific annotation rules that are not obvious from examples alone. LLMs can know biomedical facts and still fail by violating annotation hierarchy, boundary rules, relation schemas, or corpus-specific label definitions. Retrieving relevant guideline instructions at inference time helps close this gap.

### How does instruction retrieval improve biomedical annotation?

Instruction retrieval searches task-specific annotation guidelines and injects relevant rules into the LLM context. This helps the model resolve edge cases, follow corpus-specific definitions, and avoid plausible biomedical answers that are wrong under the target dataset schema.

### Why use two-step inference?

Many annotation tasks require structured output, but strict formatting can suppress useful reasoning. Two-step inference lets the model reason first, then convert the result into the required schema, preserving both interpretability and machine-readable labels.

### Which biomedical text-mining tasks are covered?

The repository covers named entity recognition, relation extraction, and multi-label text classification. The included datasets and prompt files cover BC5CDR, NCBI-Disease, BioRED, LitCovid, and Hallmarks of Cancer tasks.

### Why distill LLM annotations into BERT models?

Closed-source frontier LLMs can be expensive or impractical for production annotation pipelines. Distilling LLM-generated synthetic annotations into smaller BERT-based models tests whether LLM annotation can reduce manual labeling costs while keeping deployment lightweight.

### What is the practical recommendation?

Use frontier LLMs as guideline-aware annotator assistants rather than unconditional replacements. The most reliable setup combines instruction retrieval, explicit reasoning, schema validation, and selective human review.

## Archival and Releases

The recommended release tag for this artifact is `v1.0-arxiv-2503.03261`. If Zenodo GitHub integration is enabled for this repository, each GitHub release can be archived automatically and assigned a software DOI.
