# Host follow-up experiments, 2026-11-07

Four earlier controlled host checks did not justify a production kernel or default change. The additional Q8 controls and startup diagnostic below also preserve separate outcomes and limits. Raw captures, original failures, and later corrections stay separate. These results describe pinned experiment runtimes on one host. They do not describe current production behavior or broad browser performance.

## Native Q8 AVX-512 panel controls

The panel plan uses base commit `f4596f1d239ca84eb18255819f56684a5c5cf086`. Its code-generation gate ran in scope `kevala-q8-panel-v2-gates-20261107-r3.scope` and passed in actual CLI and probe builds, with default and global AVX2 modes. Each strict mode checked 12,028 cases and 8,697,298 values. The gate receipt SHA-256 is `6871878950bdc7b94e13f9cfbdc8074210d13fc9face3e756f455dad6cf597fe`; its source-index SHA-256 is `200919837a6f374a5355a442440b837c8b1068b36121a8e24b9421a9ca1a4029`.

The separate `native-q8-panel-avx512-v2-r3` run in scope `kevala-q8-panel-v2-panel-20261107-r3.scope` timed a synthetic finite 4-row, K=1024 Q8 conversion panel with block size 32 and 131,072 repetitions. It did not read the model pack or run inference. The 25 ms calibration protocol used nine alternating quartets and retained 18 batch observations per arm. Its paired A/B panel ratio was 2.594092, with a range of 2.420875 to 2.919546 and 9 of 9 wins. The identical-library A/A file-copy control had a median ratio of 1.022512, a range of 0.923073 to 1.121150, and 6 of 9 wins. Seventeen A/B candidate batches were below the declared 25 ms target; no baseline batches were below it.

The predeclared policy failed its A/A median and quartet checks, although the A/B median and quartet checks passed. The A/B result therefore remains an isolated conversion-panel result. It is not eligible for a real projection or production promotion. The panel summary SHA-256 is `9ea6b87a611e7b31485dc050ec187e3ca88e47965b75ea618fa10571f21e15f1`; the policy-decision SHA-256 is `a0b5c7182e45ea981d2d9b5da882c30da8758c931d96c20580571cfdae312455`.

A distinct `native-q8-panel-aa250-v1` run in scope `kevala-q8-panel-v2-aa250-20261107-r1.scope` then measured only a baseline-versus-identical-baseline control. Both arms used library SHA-256 `0edb27c57af4246235bd73525dc87fb3ad8f9692ccad3e4657616530ff964a09` at distinct device/inode identities. It used the same 4-row, K=1024 synthetic fixture, but a separate 250 ms minimum-batch protocol. All 45 post-freeze batches met that minimum; the shortest was 303.845 ms. Its nine paired ratios had a median of 1.005728 and a range of 0.712690 to 1.072887. The median check passed its predeclared 0.98 to 1.02 interval, but two quartets failed the 0.95 to 1.05 interval. The independent arm-median ratio was 0.999440. The independent data audit passed, while the control policy failed. This run does not rescue the earlier A/B result. No projection, candidate timing, or promotion followed, and the measurement line stopped. The 25 ms and 250 ms protocols remain separate; the samples do not establish a cause for the A/A spread.

The AA250 receipt SHA-256 is `6d57f30583dc347ef61e393693a4494c4ae955d621dc2fcb9ee750e789533d70`; its plan SHA-256 is `7719bec9a9d1df0698af5f82e0906cd24c2bd45cc936b3184db103a50a7ae5ea`, source-index SHA-256 is `76980f5362c7b1ae713297c56cb017e24f82ede7794932e962815b79b529b5a3`, and independent audit SHA-256 is `9feb034b726753afe8a261e6de2cc013bc978b03fb5984371f200f6393ce0f89`.

## Clef startup affinity diagnostic

The private ChromeService/session-bus startup diagnostic ran in scope `kevala-clef-affinity-startup-v1-private-sbus-r2-20261107.scope` and was stopped before browser navigation, payload access, or benchmark math. Its independent evidence audit passed validation of the retained records, while the run itself ended with an owned-tree affinity rejection. The session-bus endpoint was absent at both recorded bookends, the parent and system bus environment stayed unchanged, and the private empty directory was removed.

The original affinity snapshots passed the CPU 7 check for their observed threads, including the anchor set. TID 46706 was absent from both snapshots. A later owned-tree guard found PID 46595, TID 46706, with an allowed CPU mask of `0-31` (`ffffffff`). The recorded thread start ticks and cgroup matched across the guard's reads, but the explicit identity-recheck flag remained false because the mask assertion raised. This does not establish that the thread changed from CPU 7 to all CPUs. The actor, mechanism, and driver cause remain unknown.

The initial owned-tree snapshot covered 15 processes and 122 threads and found no app scopes. The final guard stopped at the affinity assertion after recording 10 processes and 66 threads. It did not reach the app-scope query, so `newAppScopes` is null and there is no complete final-tree pass. No model, GPU, numerical, or performance result came from this startup-only run.

The startup plan SHA-256 is `49475c0bbadb4a47d8e0cc129fe2f5d656faad2d6cfe342146476d874567b75e`, source-index SHA-256 is `fd214a72b06d8541a3bfecfb54c21def47cc790832c2f87001a23ee3e39cbf1f`, original run-receipt SHA-256 is `06af6fbef28358e01d44763f03e66c09130bd000d59a20631d0c5c3a83ddd8cd`, and independent-audit SHA-256 is `d67d3da3f6e92f5b5e185bfb37424606b5b9707fd8f65f3812eacd1e263b04cf`.

## Matmul shape constants

The original `matmul-shape-constants-v1` run failed during WebGPU device setup with `A valid external Instance reference no longer exists`. The controller had already hashed local tensor ranges, but the browser did not fetch the selected tensors or run matrix math. The failed receipt remains separate from the later diagnostic. Its SHA-256 is `b5e6178059913bce99d921fb7ebcfcad74b44821be7595186019e2574b803af6`.

A separate `matmul-shape-constants-diagnostic-v1` run passed a controlled real-tensor screen. Its receipt SHA-256 is `74967220149fdfdbb9bf37ae9f4da76924f4eb54da680929a3950cb0a1e0d59c`. Dynamic-to-constant latency ratios were 0.99302, 0.98017, and 1.00209 for Kev shapes T31, T158, and T532. Clef ratios were 0.99417, 0.96829, and 0.99786 for T158, T512, and T659. The only positive result was 0.21%, within A/A variation. The change was rejected. No shader or default changed.

The diagnostic logs still contain an `Instance` warning. The 19,520,808-byte `chromedriver.log` is not included in the publication archive. Its size is recorded in the receipt, despite the run's original warnings/errors label. This experiment did not verify a browser logging fix. A legacy summary synthesized `isFallbackAdapter=false` and did not retain the raw nullable adapter fields. Treat that value as a summary, not raw evidence.

## FP32 K32 and K64 projection cell

The `matmul-fp32-k64-v1` screen compared FP32 K32 and K64 on one real Clef Q8 layer-zero `gate_up` cell with T=512, N=24,576, K=4,096, and split=1. The independent data audit passed: timestamps were positive and ordered, summary statistics recomputed exactly, and finite-output, canary, and stability checks passed. The receipt SHA-256 is `17f6d8e9805ca48dd3816340be3b384629577f7d285090b1ec2f1424330161da`. The raw result SHA-256 is `9ad75bb15e54ee8c787f63fa0d7a78eda06cad6dfa5081f2260d734d2d9f4918`.

The median paired baseline-to-candidate ratio was 0.6222935, with 0 of 9 K64 wins. Independent medians were 5.734116 ms for K32 and 9.059576 ms for K64. The identical-K32 control had a 1.032139 median ratio and a 0.9329 to 1.0722 range. K64 was about 60.7% slower in this paired kernel screen. The predeclared Kev cell was skipped by the first-cell rule, without reading its pack payload. The result rejects K64 for this cell. It does not establish a model or application latency change, a default change, or a causal occupancy explanation.

NaN bias cases returned `0x7fffffff` in both variants. This is differential agreement only and does not establish portable WGSL NaN behavior. The captured browser metadata reported `adapter.info.isFallbackAdapter=false`; `GPUAdapter.isFallbackAdapter` was `null`. One retained INFO console message was `A valid external Instance reference no longer exists.` The check did not verify a browser warning fix. No fresh whole-pack hash was taken, and the GPU timestamp excludes uploads, compilation, readback, model execution, and application work.

The test used source commit `8d4e4fa458c8faf724d624670de721bd51cd7175` with CI75 WASM. A later metadata-only binding matched all 53 runtime asset descriptors to the actual CI76 delivery, with SHA-256 `a057048d9123d9ac1d0c848972311d74c7053e6d98cc1f26354cf42ddca50687`. The final cleanup record has SHA-256 `519ce73678cfc1b0405d22fc1b1463701c85fdc8d40b5ca77739f9a797817a77`. This binding adds provenance only. It did not add another GPU or model run.

## Gemma attention calibration budget

The cold comparison used a 750 ms baseline and a 1,500 ms candidate calibration budget. It planned 12 fresh loads, but stopped after the first E2 pair when the exact load-warmup hidden hash differed. The top-level receipt is failed and has SHA-256 `5dbb653a77dbaee7e98731c252aae3ec3b13cbf3f0233586ff1a9bdfff5f223d`. E4 was not reached.

Baseline and candidate load times were 20.347 s and 18.798 s. Their short-request medians were 21.7 ms and 25.5 ms. Their long-request medians were 75.7 ms and 80.8 ms. The candidate completed 12 tuning points but lost the baseline's local-long selection. Other matvec choices also differed, so this pair does not isolate the budget change.

All 24 paired request hashes matched. Public JSON and hidden-state hashes differed, while the rounded answers matched. The maximum public probability difference was 9.368e-19. The hidden arrays were not retained, so no hidden-state difference magnitude is available. The cold-budget change was rejected. No default changed.

A separate warm `gemma-attention-budget-v1` run reported a 3.1% long-request reduction. Its warm state confounds the comparison, so it does not establish that a longer cold calibration budget caused a speedup. Its receipt has SHA-256 `df1adf9f9d36721f4b9890042b715e2d60801997020099cb50db960eb3423353` and records source commit `dd6c94a25cb707112ff7f146ceff756d1157cfd0` with CI artifact run `42340795009`. The cold campaign records production worktree base `4e27753e6f385d6fabbb6040fdd82f058a9c48a8` and baseline CI73 run `42350139676`. Its candidate was an unpublished source overlay: it changed only `DEADLINE_MS` in `gpu-gemma4-attention.js` from 750 to 1,500 ms and used the same three CI73 WASM files. The candidate JS SHA-256 was `00b24d908fbad27ad2b5630db5faa2633c4f342ab0485bc5e29e5640e44ccca6`. These recorded identities distinguish source commits, artifact byte references, and the candidate edit. They do not establish the current runtime.

## Firefox submission modes

The Firefox capture compares `await`, `split`, and `none` on a forced-page Laya Q8 run. The original runner failed its strict derived-ratio check by one ULP. The raw timing file remains unchanged at SHA-256 `3c5c9affc1d8953f3b6805608ecbe530abe939564c77bdae3dfa09b352c7fee9`. The failed runner receipt remains SHA-256 `b884452a76312645ee8d329ebedfc95ae614eb2e26bad665481aa23a8b610d37`.

The offline helper `revalidate-ratio-drift.mjs` has SHA-256 `d875a3ed678b64af348222bc65d39bdc060164ddbbd9eff106c40341755a16d5`. It recomputes all block and case summaries from raw calls and samples, then asserts that exactly 15 changed fields are derived ratios and that every non-ratio field remains exact. The unchanged strict validator passes. The helper does not change the raw data, arm means, timings, or outputs. The corrected derived receipt is separate and has SHA-256 `f72f6465c9a1b965b75e59c6d711948a28b4351262c2b6d14cf8a23d56130ed6`.

For the 512-token request, the paired `await/split` median ratio was 5.498 with a range of 4.991 to 9.264. The `await/none` ratio was 6.665 with a range of 4.943 to 8.754. The `split/none` ratio was 1.010 with a range of 0.801 to 1.754. Both `split` and `none` beat `await` in all nine blocks. The data does not identify a reliable winner between `split` and `none`, or a consistent short-request winner. Independent per-mode medians were 566 ms for `await`, 79.5 ms for `split`, and 77 ms for `none`.

This was Firefox 152.0.3 on a historical runtime. The one-ULP derived-ratio discrepancy cannot be attributed from the retained evidence to Firefox division or WebDriver transport. The capture is not evidence about current production, UI responsiveness, or a 100 ms fix. The current per-engine submission policy remains unchanged.

## Evidence package

The [evidence package README](benchmarks/host-followup-experiments-2026-11-07/README.md) lists the fixed source and raw-data inventory, exclusions, and review commands. The archive preserves original failed receipts separately from corrected supplemental evidence. It excludes weights, compiled WASM, full copied runtime trees, and the 19.5 MB ChromeDriver log.

The [host-controls evidence package](benchmarks/host-controls-2026-11-07/README.md) preserves the Q8 panel controls, the separate AA250 baseline control, and the startup diagnostic with their complete retained timing captures and source identities. Its manifest records exact file hashes and archive hashes. The raw startup and lease-event logs are excluded, so the published independent audit is a selected report and cannot reproduce the full startup validator or thread inventory.
