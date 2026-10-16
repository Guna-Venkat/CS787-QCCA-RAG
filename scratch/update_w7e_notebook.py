#!/usr/bin/env python3
"""Update week7e_multifidelity_representation.ipynb with repaired Section 10 and Section 12."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
nb_path = PROJECT_ROOT / "notebooks/week7/week7e_multifidelity_representation.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Loaded notebook with {len(nb['cells'])} cells.")

# Section 10 Markdown Source
sec10_md = [
    "## Section 10 — TRAIN / INTERNAL-VAL Medium Cap Selection (Repaired Hybrid Sweep)\n",
    "Under the corrected **Hybrid MEDIUM** formulation:\n",
    "$$\\boxed{\\text{MEDIUM} = \\text{extractive text} + \\text{original SFR embedding}}$$\n",
    "increasing the token cap no longer replaces an embedding — all candidate caps ($M_{64}, M_{96}, M_{128}$) retain the full SFR embedding slice.\n",
    "Consequently, the information tradeoff changes: $M_{64} + \\text{SFR}$ may outperform $M_{96} + \\text{SFR}$ by introducing less distracting lexical context while preserving whole-passage semantics.\n",
    "\n",
    "We evaluate downstream answer F1 and context tokens on all 342 INTERNAL-VAL queries across:\n",
    "- $4F + 2(M_{64} + H) + 4H$\n",
    "- $4F + 2(M_{96} + H) + 4H$\n",
    "- $4F + 2(M_{128} + H) + 4H$\n",
    "\n",
    "**Predeclared Freezing Rule:**\n",
    "$$\\boxed{\\text{Highest INTERNAL-VAL } F1; \\quad \\text{within } 0.003 \\text{ tolerance choose fewer context tokens}}$$\n",
    "DEV evaluation is strictly gated until this frozen cap is established.\n"
]

# Section 10 Code Source
sec10_code = [
    "# ==============================================================================\n",
    "# Repaired INTERNAL-VAL Downstream Sweep & Freeze: Hybrid MEDIUM (M64, M96, M128)\n",
    "# ==============================================================================\n",
    "# 1. Load manifest and items\n",
    "val_manifest_path = PROJECT_ROOT / cfg[\"paths\"][\"internal_val_manifest\"]\n",
    "df_val_manifest = pd.read_csv(val_manifest_path)\n",
    "val_qids = set(df_val_manifest[\"question_id\"].astype(str))\n",
    "val_items = [item for item in train_items if str(item.get(\"question_id\", \"\")) in val_qids]\n",
    "print(f\"Loaded {len(val_items)} internal validation queries for repaired cap selection.\")\n",
    "\n",
    "# Reference canonical baselines on INTERNAL-VAL\n",
    "train_resp_path = PROJECT_ROOT / cfg[\"paths\"][\"train_response_matrix\"]\n",
    "df_train_resp = pd.read_csv(train_resp_path)\n",
    "val_resp_df = df_train_resp[df_train_resp[\"question_id\"].astype(str).isin(val_qids)]\n",
    "ref_k4_f1 = float(val_resp_df[\"f1_k4\"].mean())\n",
    "ref_k4_tok = float(val_resp_df[\"tokens_k4\"].mean())\n",
    "ref_k6_f1 = float(val_resp_df[\"f1_k6\"].mean())\n",
    "ref_k6_tok = float(val_resp_df[\"tokens_k6\"].mean())\n",
    "print(f\"Canonical INTERNAL-VAL Baselines: k=4 F1={ref_k4_f1:.4f} ({ref_k4_tok:.1f} tok) | k=6 F1={ref_k6_f1:.4f} ({ref_k6_tok:.1f} tok)\")\n",
    "\n",
    "# 2. Output and tables directories\n",
    "train_dir = PROJECT_ROOT / cfg[\"paths\"][\"train_dir\"]\n",
    "train_dir.mkdir(parents=True, exist_ok=True)\n",
    "tables_dir = PROJECT_ROOT / cfg[\"paths\"][\"tables_dir\"]\n",
    "tables_dir.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "# 3. Downstream Generation / Result Aggregation for M64, M96, M128\n",
    "results_by_cap = {}\n",
    "for cap in [64, 96, 128]:\n",
    "    cache_path = train_dir / f\"e1h_m{cap}_val_generation.jsonl\"\n",
    "    if cache_path.is_file():\n",
    "        df_cap = pd.read_json(cache_path, lines=True)\n",
    "        results_by_cap[cap] = df_cap\n",
    "        print(f\"M{cap}: Loaded {len(df_cap)} queries from cache -> Mean F1 = {df_cap['token_f1'].mean():.4f}, Mean Tokens = {df_cap['context_tokens'].mean():.1f}\")\n",
    "    else:\n",
    "        print(f\"\\nExecuting resumable generation for INTERNAL-VAL Hybrid MEDIUM (M{cap})...\")\n",
    "        def val_allocator(item: Dict[str, Any]):\n",
    "            qid = str(item[\"question_id\"])\n",
    "            return allocate_e1_primary(len(item.get(\"passages\", [])), semantic_scores=train_semantic_scores[qid])\n",
    "        df_cap = execute_resumable_multifidelity_generation(\n",
    "            sara_model=sara_model,\n",
    "            sara_tokenizer=sara_tokenizer,\n",
    "            records=val_items,\n",
    "            allocator_fn=val_allocator,\n",
    "            medium_texts_by_qid=val_medium_by_cap[cap],\n",
    "            cached_sfr_embeddings=train_sfr_dict,\n",
    "            cache_path=cache_path,\n",
    "            split=\"internal_val\",\n",
    "            policy_name=f\"E1_hybrid_4F_2M_4H_M{cap}\",\n",
    "            token_cap=cap,\n",
    "            device=device,\n",
    "            max_new_tokens=128,\n",
    "            hybrid_medium=True,\n",
    "        )\n",
    "        results_by_cap[cap] = df_cap\n",
    "\n",
    "# 4. Assemble Comparative Selection Table\n",
    "table_rows = []\n",
    "for cap in [64, 96, 128]:\n",
    "    df_c = results_by_cap[cap]\n",
    "    m_f1 = float(df_c[\"token_f1\"].mean())\n",
    "    m_tok = float(df_c[\"context_tokens\"].mean())\n",
    "    table_rows.append({\n",
    "        \"Hybrid cap\": f\"M{cap}\",\n",
    "        \"F1\": round(m_f1, 4),\n",
    "        \"Tokens\": round(m_tok, 1),\n",
    "        \"ΔF1 vs k4\": round(m_f1 - ref_k4_f1, 4),\n",
    "        \"Δtokens vs k4\": round(m_tok - ref_k4_tok, 1),\n",
    "        \"ΔF1 vs k6\": round(m_f1 - ref_k6_f1, 4),\n",
    "    })\n",
    "\n",
    "df_hybrid_cap_selection = pd.DataFrame(table_rows)\n",
    "table_out_path = tables_dir / \"w7e_hybrid_cap_selection.csv\"\n",
    "df_hybrid_cap_selection.to_csv(table_out_path, index=False)\n",
    "\n",
    "print(\"\\n\" + \"=\" * 75)\n",
    "print(\"REPAIRED INTERNAL-VAL HYBRID MEDIUM TOKEN CAP SELECTION TABLE\")\n",
    "print(\"=\" * 75)\n",
    "display(df_hybrid_cap_selection)\n",
    "\n",
    "# 5. Apply Predeclared Freezing Rule: highest F1; within .003 choose fewer tokens\n",
    "max_f1 = df_hybrid_cap_selection[\"F1\"].max()\n",
    "within_eps = df_hybrid_cap_selection[df_hybrid_cap_selection[\"F1\"] >= max_f1 - 0.003]\n",
    "winner_row = within_eps.sort_values(\"Tokens\", ascending=True).iloc[0]\n",
    "FROZEN_HYBRID_MEDIUM_CAP = int(winner_row[\"Hybrid cap\"].replace(\"M\", \"\"))\n",
    "FROZEN_MEDIUM_CAP = FROZEN_HYBRID_MEDIUM_CAP\n",
    "\n",
    "print(\"\\n\" + \"=\" * 75)\n",
    "print(f\"--> FROZEN HYBRID MEDIUM TOKEN CAP WINNER: M_{FROZEN_HYBRID_MEDIUM_CAP} <--\")\n",
    "print(f\"Selected via rule: highest F1 ({max_f1:.4f}); within 0.003 tolerance chose lowest tokens ({winner_row['Tokens']})\")\n",
    "print(\"=\" * 75)\n",
    "\n",
    "# Save selection metadata\n",
    "frozen_meta = {\n",
    "    \"frozen_hybrid_cap\": FROZEN_HYBRID_MEDIUM_CAP,\n",
    "    \"selection_rule\": \"highest F1 on INTERNAL-VAL; within 0.003 choose fewer tokens\",\n",
    "    \"max_f1\": float(max_f1),\n",
    "    \"summary\": df_hybrid_cap_selection.to_dict(orient=\"records\"),\n",
    "}\n",
    "with open(tables_dir / \"frozen_hybrid_cap.json\", \"w\", encoding=\"utf-8\") as f:\n",
    "    json.dump(frozen_meta, f, indent=2)\n"
]

# Section 12 Code Source
sec12_code = [
    "# Free any lingering allocations before loading SARA\n",
    "import gc\n",
    "gc.collect()\n",
    "if torch.cuda.is_available():\n",
    "    torch.cuda.empty_cache()\n",
    "\n",
    "# Assert repaired INTERNAL-VAL cap selection was completed and frozen\n",
    "tables_dir = PROJECT_ROOT / cfg[\"paths\"][\"tables_dir\"]\n",
    "assert \"FROZEN_HYBRID_MEDIUM_CAP\" in globals() or (tables_dir / \"frozen_hybrid_cap.json\").is_file(), (\n",
    "    \"Section 10 must be executed to freeze FROZEN_HYBRID_MEDIUM_CAP before evaluating DEV!\"\n",
    ")\n",
    "if \"FROZEN_HYBRID_MEDIUM_CAP\" not in globals():\n",
    "    with open(tables_dir / \"frozen_hybrid_cap.json\", \"r\") as f:\n",
    "        FROZEN_HYBRID_MEDIUM_CAP = json.load(f)[\"frozen_hybrid_cap\"]\n",
    "FROZEN_MEDIUM_CAP = FROZEN_HYBRID_MEDIUM_CAP\n",
    "\n",
    "print(f\"Verified Frozen Hybrid Medium Cap for DEV: M_{FROZEN_HYBRID_MEDIUM_CAP}\")\n",
    "\n",
    "# Load SARA model for inference\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "print(f\"Loading SARA model for evaluation on {device}...\")\n",
    "sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=device)\n",
    "\n",
    "dev_gen_cache_e1_hybrid = PROJECT_ROOT / cfg[\"paths\"][\"dev_dir\"] / \"e1_hybrid_generation.jsonl\"\n",
    "\n",
    "# Allocator closure passing precomputed semantic scores\n",
    "def e1_hybrid_allocator(item: Dict[str, Any]) -> Tuple[List[int], List[int], List[int]]:\n",
    "    qid = str(item[\"question_id\"])\n",
    "    s_scores = dev_semantic_scores[qid]\n",
    "    return allocate_e1_primary(len(item.get(\"passages\", [])), semantic_scores=s_scores)\n",
    "\n",
    "print(f\"Executing resumable generation for E1-H Hybrid Policy (token_cap={FROZEN_HYBRID_MEDIUM_CAP})...\")\n",
    "df_dev_e1_hybrid = execute_resumable_multifidelity_generation(\n",
    "    sara_model=sara_model,\n",
    "    sara_tokenizer=sara_tokenizer,\n",
    "    records=dev_items,\n",
    "    allocator_fn=e1_hybrid_allocator,\n",
    "    medium_texts_by_qid=dev_medium_by_cap[FROZEN_HYBRID_MEDIUM_CAP],\n",
    "    cached_sfr_embeddings=dev_sfr_dict,\n",
    "    cache_path=dev_gen_cache_e1_hybrid,\n",
    "    split=\"dev\",\n",
    "    policy_name=\"E1_hybrid_4F_2M_4H\",\n",
    "    token_cap=FROZEN_HYBRID_MEDIUM_CAP,\n",
    "    device=device,\n",
    "    max_new_tokens=128,\n",
    "    hybrid_medium=True,\n",
    ")\n",
    "\n",
    "# Strict invariant assertion on all 231 DEV rows\n",
    "for _, row in df_dev_e1_hybrid.iterrows():\n",
    "    assert_multifidelity_row_invariants(\n",
    "        row,\n",
    "        n_full_expected=4,\n",
    "        n_medium_expected=2,\n",
    "        n_heavy_expected=4,\n",
    "        hybrid_medium=True,\n",
    "    )\n",
    "\n",
    "e1h_f1_dev = float(df_dev_e1_hybrid[\"token_f1\"].mean())\n",
    "e1h_tokens_dev = float(df_dev_e1_hybrid[\"context_tokens\"].mean())\n",
    "print(f\"\\nPolicy E1-H (4F + 2(M+H) + 4H) on DEV [M_{FROZEN_HYBRID_MEDIUM_CAP}]: Mean F1 = {e1h_f1_dev:.4f}, Mean Tokens = {e1h_tokens_dev:.1f}\")\n",
    "print(f\"Delta over E0: Delta F1 = {e1h_f1_dev - e0_f1_dev:+.4f}, Delta Tokens = {e1h_tokens_dev - e0_tokens_dev:+.1f}\")\n"
]

# Locate Section 10 and Section 12 cells
sec10_idx = None
sec12_idx = None

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        src = "".join(cell.get("source", []))
        if "Section 10 — TRAIN / INTERNAL-VAL Medium Cap Selection" in src:
            sec10_idx = i
        elif "Section 12 — E1-H Primary Policy" in src:
            sec12_idx = i

print(f"Found Section 10 at cell {sec10_idx}, Section 12 at cell {sec12_idx}")

if sec10_idx is not None:
    nb["cells"][sec10_idx]["source"] = sec10_md
    nb["cells"][sec10_idx + 1]["source"] = sec10_code
    nb["cells"][sec10_idx + 1]["outputs"] = []
    nb["cells"][sec10_idx + 1]["execution_count"] = None

if sec12_idx is not None:
    nb["cells"][sec12_idx + 1]["source"] = sec12_code
    nb["cells"][sec12_idx + 1]["outputs"] = []
    nb["cells"][sec12_idx + 1]["execution_count"] = None

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Successfully updated notebook cells for Section 10 and Section 12!")
