# Demo reproducibility status — final freeze

Git-indexed demo files: **85**. Source, tests, UI/CSS, tokenizer-only assets, documentation, pinned environment snapshot, publication metadata and the versioned curated archive are preserved on branch `gagan`. The user-authorized final freeze replaces the prior blanket `/demo/` ignore with selective ignores.

## Clean checkout and immutable data restore

`demo/releases/curated-demo-v1.tar.gz` is the versioned immutable archive mechanism, stored alongside source. It is 13.93 MiB compressed and contains the original bundle, publication receipt, curated manifest and SHA256SUMS. The raw approximately 98 MiB JSON is ignored; it is not an ordinary mutable Git object. Git LFS or external hosting is not required.

```bash
python3.11 -m venv demo/.venv
demo/.venv/bin/python -m pip install -r demo/requirements.txt
demo/.venv/bin/python demo/restore_curated_bundle.py
demo/.venv/bin/python -m streamlit run demo/app.py
```

The restore checks archive/member checksums before writing, refuses incompatible existing files and is idempotent. The demo can read the Git-tracked compressed split-metadata JSON without recreating an uncompressed research artifact. Optional live model weights/checkpoints remain external; curated playback needs no model download or inference.

## Intentionally excluded

Environments, model weights, checkpoints, credentials, Streamlit/runtime caches, arbitrary user inputs, debug logs, temporary generations and large intermediate build shards remain ignored. The explicitly approved static World Cup diagnostic is a versioned preset fixture; it is not a collection of arbitrary user session input. Its frozen report is documentation, while temporary condition shards/cache remain excluded.

## File integrity inventory

| Path | Git indexed? | Bytes | SHA256 |
|---|---|---:|---|
| `demo/.gitignore` | Yes | 59 | `e541c32e9063d49d986014c983fcace76596ca3c697412be6cc2d6682613d3f8` |
| `demo/APPLICATION_GUIDE.md` | Yes | 59530 | `af12a18a183f7e2c2a93a5ef244ee2d563278d75720629c23020feb23c521fd9` |
| `demo/FINAL_FREEZE_STATUS.md` | Yes | 1765 | `803bcf21fd3c1245f185f60711bcd4d36b34b2bf1fc46a0a3ae457e316722eb3` |
| `demo/IMPLEMENTATION_REPORT.md` | Yes | 21411 | `0792e14f0eab00a2ea524919caa6b6a4944ce3abb7be1c106e74094b65f95e97` |
| `demo/README.md` | Yes | 22139 | `1aa72a476eea6b2ea3e5b67064f8604357d98bc3de5baf1049cb63d08083b95f` |
| `demo/__init__.py` | Yes | 71 | `63e22e4d97ef69d5b949a7c43fa134e162253bc7ccb8bd3d8f34fc3e7617c895` |
| `demo/app.py` | Yes | 30743 | `8d8b372251e343f3e0b400bc6542f432086c36644c1d94261458bb3d9e0e4dd9` |
| `demo/artifact_manifest.json` | Yes | 6244 | `9acd4ee42de11a04957686ecb46a2b028781f76f93f5499a76cce46fc71a6607` |
| `demo/assets/live_tokenizer/added_tokens.json` | Yes | 707 | `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` |
| `demo/assets/live_tokenizer/chat_template.jinja` | Yes | 4168 | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `demo/assets/live_tokenizer/merges.txt` | Yes | 1671853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `demo/assets/live_tokenizer/source.json` | Yes | 817 | `52f056d2794b98733f1a15df9357d86498ed45078522d88dd6e0ceb813191e34` |
| `demo/assets/live_tokenizer/special_tokens_map.json` | Yes | 613 | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| `demo/assets/live_tokenizer/tokenizer.json` | Yes | 11422654 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `demo/assets/live_tokenizer/tokenizer_config.json` | Yes | 5404 | `443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0` |
| `demo/assets/live_tokenizer/vocab.json` | Yes | 2776833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `demo/assets/presets/world_cup.json` | Yes | 9944 | `2918d3bfe087fb785a8476a8d0c339331cdde89b11339d5dfa353e4fbf55bbc8` |
| `demo/assets/styles.css` | Yes | 4532 | `f093dbb82f1e197f5aa9bb33ee20a5048741cc7a6d9fa6443e2e0090a9c89687` |
| `demo/audit/FINAL_DEMO_CORRECTNESS_AUDIT.md` | Yes | 19319 | `ad5b36b75670ced31fc3164ea0b19204e681eb9035512a999cb9ae550d7d5fb1` |
| `demo/audit/audit_qasper_splitter.py` | Yes | 7943 | `9c3fed4809e5fa4c13c6f2e94ac583a3a7ddb5170a8b517fb5af7fe6d0456ddc` |
| `demo/audit/audit_reproducibility.py` | Yes | 4532 | `38cc56916895e258649e5b60ecdeaa2ddd790cb2086dda5a56bb5231f4358c7f` |
| `demo/audit/clean_checkout_verification.json` | Yes | 296 | `14647b03ae5ff9fd997f11edddb1bd26669d263410d4d37b945d4aefcfb6507c` |
| `demo/audit/correctness_protected_before.json` | Yes | 111371 | `b7ecc07c5a27c33f6a14084a0789c9fef5fb059c602aaba7f71fcc08dd6f8282` |
| `demo/audit/final_methodology_audit.md` | Yes | 19319 | `ad5b36b75670ced31fc3164ea0b19204e681eb9035512a999cb9ae550d7d5fb1` |
| `demo/audit/live_sentence_segmentation_audit.md` | Yes | 52426 | `15902b4c56eea0a7a88690dc1a6bc1da1b838bdbcfbbe63f90307b1535c58537` |
| `demo/audit/protect_artifacts.py` | Yes | 1248 | `6394fa1ef4d0d8a992da25c774790611bf1ecf8fcf4875dc0ddf4ff97d5246ff` |
| `demo/audit/qasper_frozen_splitter_impact.md` | Yes | 5198 | `f197147076ecffb5df65aad10b1b58c35a9e6f1ab482a6f417401c2873558afe` |
| `demo/audit/qbgf_score_scale_audit.md` | Yes | 3779 | `d647bc85fec4b99ed3897f9a8961f52376b9e2333f1c6c7df65e68d4f05df221` |
| `demo/audit/refine_qasper_impact.py` | Yes | 3079 | `4ec43e5cc082cdd6cc66779ee44bcca7a8f76c5048b784f7f64ecb0240c801b2` |
| `demo/audit/requirements_qualified_freeze.txt` | Yes | 4577 | `dfa6c67e3e7e2615bb680f8f7d6de3d77f5fedb185185a50e85676200a9c5184` |
| `demo/audit/run_sentence_segmentation_audit.py` | Yes | 11953 | `3cc2ee68222c21f0ae4396629435ebe0e8a779e41b7e657c466e1043d9ae9e3d` |
| `demo/audit/run_world_cup_correctness.py` | Yes | 8889 | `0a96cd24c5f578b21ff9e0b9305f0a2582b778509042a5dc18c831b6628e7599` |
| `demo/audit/score_scale_report.py` | Yes | 3549 | `f934dfc30e80766ba8b5ca12bfb4a4421fa39f68fa67dce15a33da4698416a33` |
| `demo/audit/world_cup_correctness_v2/COMPLETE.json` | Yes | 161 | `3e818f28bbbcaee9a84daee2b414ed6c5f2e1e56cc60876f1d8989414b0b1327` |
| `demo/audit/world_cup_correctness_v2/WORLD_CUP_12_CONDITION_REPORT.md` | Yes | 48806 | `5e5a498a690b75d37375c3938bdd46146204cb5096131edd5bc382a716f66802` |
| `demo/audit/write_correctness_report.py` | Yes | 20402 | `70d8265f9d779b6b94b6f1cf49a6b67a967ee07d5ffcb15bb3fcdf7722e28698` |
| `demo/build_curated_replays.py` | Yes | 183 | `359e58e702fc68422f7a8f24c24aa232c919272f61f86c2a34e0d618241d042a` |
| `demo/components.py` | Yes | 3156 | `33783a7e02e093e3ac84809e96233ea644f91a1f7999185552c0e4623ed13e07` |
| `demo/constants.py` | Yes | 4458 | `c8f47c2c33d94b19fed1c05025e52551c4c16a23c4e6e6e5c36daeab8b07860d` |
| `demo/curated/__init__.py` | Yes | 89 | `324c61404710dcd25aa8a18e0243642d0e8316f98f6d5e9ee67e1dcd3cb7cc7a` |
| `demo/curated/cli.py` | Yes | 1603 | `474626a8f88090f44a4dd6c44f1674bace7e7abc7e6d39e61ab6e107e292dde5` |
| `demo/curated/data.py` | Yes | 980 | `566394989dd0a8f5acde0a72c83b7924013ce0971a6216fe2df308dfd806e87e` |
| `demo/curated/generation.py` | Yes | 3930 | `b95f6e96582adb61178ee5e7016c4ad0c90451a3026c4bf533280a779f232ce2` |
| `demo/curated/preparation.py` | Yes | 5145 | `68cc949071e5b794e96b7a082cf9329c7160802c5e1d1aa6ab9bec3e0ee5cc36` |
| `demo/curated/protocol.py` | Yes | 4791 | `6ddbe71417e5c60aa7e9d4eacfce224789f62e8aa55362551de7c99dc680b6f9` |
| `demo/curated/publication.py` | Yes | 4361 | `7b402b410ec152d6c1c3310e3c69c9e5c04ab2f803a9786d5be42a03f88cb198` |
| `demo/curated/ui.py` | Yes | 9830 | `f7d0b84ab6cc32fcd5198f52abf621b29425b37375c40230e015d2653c0e4f7c` |
| `demo/data_loader.py` | Yes | 4453 | `a5b43c36016297461b012a6027a6d982ecfef5f4cdb6111c92c31fa6085bff12` |
| `demo/evidence_registry.py` | Yes | 5668 | `df50bc3c80fd5487317420e6497e991dac9d2b35a8e61b5e60ed25c25b1918e7` |
| `demo/example_catalog.json` | Yes | 1341 | `a8a05747cf6ea4f9a292f4501f5b9f11cbfe2d2d98fb000d43ad71358bef347b` |
| `demo/live/__init__.py` | Yes | 66 | `ef58552bd0ba012d7574a5d7fb5ffcdc14edcbaba5c0d19a0b6e1a6b5dd93804` |
| `demo/live/engine.py` | Yes | 16144 | `dbf6332bbc676c04babbf61d8e17f920d7f99ca61000e2fe4314ea08603a3936` |
| `demo/live/preset.py` | Yes | 1521 | `2895990eacae27d8d0d507cd6a0f543df37a6a1eaec6fade8768826fc51603f1` |
| `demo/live/protocol.py` | Yes | 2679 | `56ef6ef8da13c83c786201cf00744391b4b010210f37383887d1a5d3f7a2a9e6` |
| `demo/live/resources.py` | Yes | 10145 | `d754783e14f70b927b23652795ae862733990d967a6661b6a6fdb7b215e163ae` |
| `demo/live/segmentation.py` | Yes | 4507 | `784eea55d1097816355064ed4f6b69bfcf5402e775c9c0885d633ed2b9fefe4a` |
| `demo/live/ui.py` | Yes | 26134 | `a7b7b1dfcdcf44ec278741a233db892eddb927188297ba5221ed3c93b6eb5c8b` |
| `demo/live_pipeline.py` | Yes | 1405 | `b9f152acd4ad310b711ceb0eb9f8620fdfe13b86e34d847e574dc2dd6bf742ff` |
| `demo/notebooks/BUILD_CURATED_REPLAYS.ipynb` | Yes | 6874 | `672dea4508e38ba85f5b3f26f57f82d017ec08adca037efc6c47fbc6b5e2f1c0` |
| `demo/plots.py` | Yes | 2206 | `b0bfdabf7e9fb28afa6b4a5096c28f801da22bc3801740c8507e9cdba695f517` |
| `demo/prepare_replay.py` | Yes | 5238 | `469535a0a6791d31e87774a1e19b777a9b651169f49a405206910886c3f700d4` |
| `demo/releases/README.md` | Yes | 1750 | `280e589c4104c635e561e5e34cc71a7d343b30102be06ae027ff5cf384f668e6` |
| `demo/releases/curated-demo-v1.archive.json` | Yes | 728 | `91ddba6926d9f51868f3c9fa04e3db532f61c33ae8ce901d8e542597a92e9ea2` |
| `demo/releases/curated-demo-v1.tar.gz` | Yes | 14605553 | `51361cdded3087f7706e9f3ae3804c19bec93f2d4a78a408595426a0359fee38` |
| `demo/replay_data/v1/PUBLISHED.json` | Yes | 237 | `2887d12a7dc47cf4ccd1a2a988815cc77c20afa27cf833240d5f9d2b3ac6d5d6` |
| `demo/replay_data/v1/manifest.json` | Yes | 8480 | `766058f888eba83617ba171a09e30bccaa899ed08722d5404783da973132f1fb` |
| `demo/requirements-test.txt` | Yes | 34 | `6cd911f8bfb31f87c2ae57e3df456617ddb9675c7a72fba2649f65d2295f5a74` |
| `demo/requirements.txt` | Yes | 32 | `042e540e5b624c7249869851a6fcbef7ff183a4fd477cd8dc6ab36697dfe9d8b` |
| `demo/research_extensions.py` | Yes | 18016 | `ea9155e85d11fbe3a870c0429c308de6a5399deb2278367b17d011aaec320063` |
| `demo/restore_curated_bundle.py` | Yes | 2617 | `a78752be5a2c5ee66f9985168d6fcde6aa24508ec95a70a237ce289fd41230e0` |
| `demo/result_registry.py` | Yes | 4733 | `fb5acfc89eff2ffb95ad2fd8c189492f628258b54bf89442e659e673854eca22` |
| `demo/setup.sh` | Yes | 535 | `3967e2e270642443fc798e0f1d57aa94bf4f9219e0b046c9f1e4dfa8b9098f3d` |
| `demo/tests/browser_live_smoke.py` | Yes | 3778 | `801514b7d06420e5b88b6a88f4ed0a961b933be4780406090c383bd894584235` |
| `demo/tests/browser_smoke.py` | Yes | 4884 | `526f95e85c6aa0ca00722c62ffe0da4c3f7e0504b3566fb7705395803eb5e34f` |
| `demo/tests/test_app.py` | Yes | 5417 | `0a2e15f0d42d732d6bb02b866dd76aba749388be0e784e9da5ffa679fda11293` |
| `demo/tests/test_curated.py` | Yes | 8399 | `3ecbbdb7d6996ef251cb36b1b7931ef97eaf6a18872827e987224ec344ca46ac` |
| `demo/tests/test_extensions.py` | Yes | 4079 | `71e6f3368737fcc952c1970d3e5eba1301c5cd5b9eba13fbb2596148c0d3ef78` |
| `demo/tests/test_final_freeze.py` | Yes | 4945 | `31e36c55feee2e7267f6199756b047448342897f383ea9c95a54c8134a607770` |
| `demo/tests/test_live.py` | Yes | 14997 | `e7ba93f57d71654377714b35e526b32678ff634ccd17bab56569edc72c3fc10b` |
| `demo/tests/test_live_correctness.py` | Yes | 6316 | `b1d530105107f9168ecb6a9137661798bf6fdc628ea5238c97069b9831b14df0` |
| `demo/tests/test_live_segmentation.py` | Yes | 4414 | `9a96de71e2aa140fc43bcc5de834440a7843600930a460b6a118afa207b3ab5f` |
| `demo/tests/test_registry.py` | Yes | 5013 | `f7e7c5525c3354d6c9aaff83625df912797d2b05e890da70f33779b82f0ed621` |
| `demo/version.py` | Yes | 114 | `e8f9ab68f9d50fa50f4f588aaf10c893361ffc553d53c5ab6c8d1c87db340931` |
| `demo/replay_data/v1/bundle.json` | Archive member | 101962674 | `14b4ba43d4cf3ab7359b54f2fc8bfe0ee59166c12f40c09125658c1d63f03036` |

## Submission procedure

Keep the versioned archive and its .archive.json checksum file unchanged; future data require a new version. Submit or push the source commit containing these files. Verify an isolated checkout by restoring the archive and launching Curated Demo. Do not rebuild the curated model-generation grid for playback. Keep a separate institutional backup of the Git release if required by submission policy. A normal clean checkout plus documented restore is sufficient for the UI; optional live inference additionally needs the locally accessible pinned models and qualified runtime.

`demo_preservation_manifest.json` is the machine-readable inventory. Derived inventory files omit their own recursive hashes. `requirements_qualified_freeze.txt` records qualification versions; `demo/requirements.txt` is the smaller curated-UI dependency entry point.
