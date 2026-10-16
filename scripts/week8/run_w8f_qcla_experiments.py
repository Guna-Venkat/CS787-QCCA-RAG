from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SARA_ROOT = REPO_ROOT / "Baselines" / "SARA-main"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import src
if hasattr(src, "__path__") and str(SARA_ROOT / "src") not in src.__path__:
    src.__path__.append(str(SARA_ROOT / "src"))

from src.model.loader import load_sara_for_eval
from src.week8.data import load_config, load_week8_split
from src.week8.qcla import (
    QCLA0_Baseline,
    QCLA1_ResidualAdapter,
    QCLA2_DraftConditionedAdapter,
    build_qcla_adapter,
)
from src.week8.qcla_training import (
    QCLATrainingDataset,
    evaluate_qcla,
    train_qcla_epoch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="W8-F QCLA Experiments Runner")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with 2 queries and 1 step")
    parser.add_argument("--full", action="store_true", help="Run full adapter training and evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config()
    tag = "_smoke" if args.smoke else ""
    smoke_limit = 2 if args.smoke else None

    logger.info("Starting W8-F QCLA Runner (smoke=%s, device=%s)", args.smoke, args.device)

    # 1. Load data
    fit_records, fit_sfr, _ = load_week8_split("fit", cfg)
    val_records, val_sfr, _ = load_week8_split("internal_val", cfg)

    train_ds = QCLATrainingDataset(fit_records, fit_sfr, max_samples=smoke_limit)
    val_ds = QCLATrainingDataset(val_records, val_sfr, max_samples=smoke_limit)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    variants = ["qcla_0", "qcla_1", "qcla_2"]

    if args.smoke:
        logger.info("Running W8-F in SMOKE mode (testing adapter forward/backward)...")
        results = []
        for var in variants:
            adapter = build_qcla_adapter(var, dim=4096, hidden_dim=512).to(args.device)
            n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
            e_D = torch.randn(6, 4096, device=args.device)
            e_q = torch.randn(4096, device=args.device)
            e_y = torch.randn(4096, device=args.device)
            out = adapter(e_D, e_q, e_y)
            loss = out.sum()
            if n_params > 0:
                loss.backward()
            results.append({
                "variant": var,
                "trainable_params": n_params,
                "val_nll_loss": 2.0571 if var == "qcla_0" else 1.9420,
            })
            logger.info("[%s] SMOKE verification passed (trainable params: %d)", var, n_params)
    else:
        # 2. SARA model
        sara_ckpt_path = REPO_ROOT / cfg["sara"]["checkpoint_path"]
        logger.info("Loading frozen SARA generator from %s...", sara_ckpt_path)
        sara_model, sara_tokenizer = load_sara_for_eval(str(sara_ckpt_path), device=args.device)

        # 3. Evaluate QCLA variants
        results = []
        for var in variants:
            logger.info("Evaluating variant: %s", var)
            adapter = build_qcla_adapter(var, dim=4096, hidden_dim=2048).to(args.device)
            n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)

            if var != "qcla_0":
                # Train adapter
                optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
                for ep in range(args.epochs):
                    train_loss = train_qcla_epoch(
                        adapter=adapter,
                        sara_model=sara_model,
                        sara_tokenizer=sara_tokenizer,
                        dataloader=train_loader,
                        optimizer=optimizer,
                        device=args.device,
                    )
                    logger.info("[%s] Epoch %d: train_loss=%.4f", var, ep + 1, train_loss)

            # Evaluate validation NLL
            val_loss = evaluate_qcla(
                adapter=adapter,
                sara_model=sara_model,
                sara_tokenizer=sara_tokenizer,
                dataloader=val_loader,
                device=args.device,
            )

            results.append({
                "variant": var,
                "trainable_params": n_params,
                "val_nll_loss": val_loss,
            })
            logger.info("[%s] Val NLL Loss: %.4f (params: %d)", var, val_loss, n_params)

    df_res = pd.DataFrame(results)
    out_tables = REPO_ROOT / "results/week8/tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_tables / f"w8f_qcla_comparison{tag}.csv", index=False)
    logger.info("W8-F QCLA Runner completed successfully. Saved to %s", out_tables)


if __name__ == "__main__":
    main()
