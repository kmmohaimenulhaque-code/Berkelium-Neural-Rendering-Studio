"""Every external model, dataset and code revision the universal-cadquery pipeline uses (CLAUDE.md rule 5).

Revisions were read from the Hugging Face / GitHub APIs on 2026-10-17. Licences are as the source states them;
"not stated" means the source repo has no licence field and must be checked before any commercial use."""

CORE = {"repo": "https://github.com/kmmohaimenulhaque-code/BERKELIUM-NEURAL-RENDERING-AND-SCIENCES",
        "branch": "physics/p1-numerical", "commit": "737dc35c42aaf97f52a6ac984a806adef9518525"}
GSPLAT_COMMIT = "ff8bc5489545bcc1b5a2b997d975834429a0d7af"

MODELS = {
    "Qwen/Qwen3-32B": {"revision": "9216db5781bf21249d130ec9da846c4624c16137", "license": "apache-2.0"},
    "Qwen/Qwen3-14B": {"revision": "40c069824f4251a91eefaf281ebe4c544efd3e18", "license": "apache-2.0"},
    "Qwen/Qwen3-0.6B": {"revision": "c1899de289a04d12100db370d81485cdf75e47ca", "license": "apache-2.0"},
}

SOURCES = {
    # arXiv 2505.06507. The paper's GitHub (Text-to-CadQuery/Text-to-CadQuery @ 4f7f501) links
    # huggingface.co/ricemonster/NeurIPS11092, which redirects to this model repo: data/data_{train,val,test}.jsonl
    # ({"input", "output"}, 90/5/5) built from Text2CAD (DeepCAD shapes) with Gemini 2.0 Flash.
    "text2cadquery": {"hf_repo": "ricemonster/IEEE-T-ASE", "repo_type": "model",
                      "revision": "4091ce30ad85b774a9231484f9b40372dc0451d3",
                      "files": ["data/data_train.jsonl", "data/data_val.jsonl", "data/data_test.jsonl"],
                      "license": "not stated (derived from Text2CAD / DeepCAD)", "paper": "arXiv:2505.06507"},
    # arXiv 2412.14042. ~1M procedurally generated CadQuery programs, train/batch_*/N.py + val/; no text.
    "cad_recode": {"hf_repo": "filapro/cad-recode-v1.5", "repo_type": "dataset",
                   "revision": "4696421935de9e9a71f1e092d8eee8b341d04619", "license": "cc-by-nc-4.0",
                   "paper": "arXiv:2412.14042"},
}

HELDOUT_FAMILIES = ("pipe_elbow", "chair", "knob", "hinge_leaf")
HF_ADAPTER = "Mhaquehaque/berkelium-qwen3-{size}-cadquery-v1"
