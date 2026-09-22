# Controlled contrasts for a π0.5 transcoder

One Colab cell for a controlled version of the layer-5 permanence experiment. The task object plays the role of the apple: arm pose, language, and noise seed stay fixed, and each condition changes one visible thing.

Paste the code cell from `controlled_contrasts.ipynb` into the existing Colab runtime, after setup has written `scripts/collect_layer5_replay.py`. It does not overwrite `layer5_replay` or the intervention-fidelity run.

## Why this is faster to read

The fidelity run on present-versus-removed pairs closed about 93–98% of the action gap with a full token patch, about 56–86% with token structure, and about 3–11% with a pooled SAE. The difference that moves the action is which tokens change. This cell trains a **per-token TopK transcoder** on layer 5 (and predicts layer 6 when that module exists), then circuit-traces only the features that track one contrast.

Natural `held_hidden` labels produced no frames in the earlier audit. Occlusion is made by editing the image instead of waiting for the gripper to hide the object.

## Conditions

At one frozen simulator state:

| Condition | What changes |
| --- | --- |
| `base` | Original render |
| `recolor` | Object color only. The pixels that change are the object mask |
| `absent` | Object teleported away. Arm qpos and proprioception stay fixed |
| `occluded` | Gray paint on half of the recolored pixels. Every other pixel matches `base` |
| `slab_miss` | That same gray shape, moved off the object |
| `occluded_absent` | The occlusion paint applied to the absent render |

A full-cover check paints the whole object mask on both the present and absent images. When those images agree and the action chunks agree, one forward of π0.5 has no remaining evidence the object is behind the cover. A small leftover action RMSE on agreeing images is the residual floor for near-matched inputs. The identical-input re-forward is the determinism check. The cell records the cover result and continues into training and circuit tracing. Permanence is the absence contrast (object removed, pose fixed) set against recolor and partial occlusion, which still leave a visible difference.

## What to read

The cell prints three blocks:

1. **Pixels** — fraction of the agent view that changed. Occlusion and `slab_miss` should be small and matched. Absence is larger.
2. **Features** — transcoder codes on the tokens that moved. A color feature should score on `color` and stay quiet on absence, occlusion, and the off-object slab.
3. **Circuit** — fraction of the action-chunk RMSE closed by patching those features from base toward the edit. `random_remap` and `off_contrast` are the same size. `full_tokens`, `pooled`, and `token_structure` keep their natural size, so they are the reference, not a matched ranking.

Outputs land in `outputs/permanence/controlled_contrasts/` (`summary.json`, `contact_sheet.png`, `transcoder.pt`). Open the contact sheet before trusting a feature.

## Knobs

Set these in the Colab cell before it launches the script. Defaults are a short run: 2 episodes, 24 steps, 4 probe frames, 512 features, TopK 16, 400 training steps.

- `CTRL_MAX_EPISODES`, `CTRL_MAX_STEPS`, `CTRL_PROBE_FRAMES`
- `CTRL_N_FEATURES`, `CTRL_K`, `CTRL_TRAIN_STEPS`, `CTRL_TOP_FEATURES`

Raise `CTRL_TRAIN_STEPS` if layer-5 R² on the probe frames is near zero. The probe episode is held out of training.

## Feature-count sweep

After the contrast cell has written `outputs/permanence/controlled_contrasts/transcoder.pt`, paste the code cell from `feature_count_sweep.ipynb`. It reloads that transcoder, rebuilds only the saved probe frames, and patches the top 8, 32, 128, and all 512 features. Each count includes a same-size random remap and an off-contrast patch. Full-token, pooled, and token-structure rows are the reference lines.

Read the color and absence verdicts under the table. A climb toward the token-structure line means the action is in the dictionary. A flat line near the 8-feature result means the next experiment is a token-position patch. The sweep writes `outputs/permanence/feature_count_sweep/` and leaves the contrast run in place.

## Token-position patch

After the contrast cell, paste the code cell from `token_position_patch.ipynb`. It rebuilds the same probe frames and adds the real per-token layer-5 delta on the top 1%, 5%, 15%, and 50% of tokens by movement. Each fraction has a same-count random-position patch and a least-moved patch. Full tokens and token structure are the reference lines. `token_l2` is the share of the token-delta norm sitting on the patched positions.

A small fraction that reaches the token-structure line means the action is in the tokens that move most. A closure that rises with the fraction means the effect is spread across the prefix. Results go to `outputs/permanence/token_position_patch/`.

## Token subspace

After the token-position cell, paste the code cell from `token_subspace.ipynb`. On the top 5% and 15% of moving tokens it patches the full per-token delta, the mean of that delta, and rank-1, rank-4, and rank-16 reconstructions. It also prints which decile of the prefix those tokens occupy, and image versus language when the vision position embedding gives a square patch count. Results go to `outputs/permanence/token_subspace/`.

## Occlusion features

`final_smaj.ipynb` is the Colab run. After the contrast cell has written `outputs/permanence/controlled_contrasts/transcoder.pt`, the occlusion cell reloads that transcoder and keeps a feature only when it rises for gray paint on the bowl and stays quiet for the same paint off the bowl, the color swap, and removal. It patches the features that pass and compares them with a same-size random patch and with the color features. Results go to `outputs/permanence/occlusion_features/`.

The paper-gaps cell keeps that same rule, scans the demonstrations for gripper contact, and writes `outputs/permanence/paper_gaps/`.

## Cover autopsy

After the paper-gaps cell, the cover-autopsy cell rebuilds those frames and splits the full cover by camera. It also records leftover pixels, a second forward of the same covered image, the first 10 actions against the rest of the chunk, arm motion against a zero action, and the depth-gap and angle-ratio distribution on grasped frames. Results go to `outputs/permanence/cover_autopsy/`.

## Paper claim

The paper-claim cell in `final_smaj.ipynb` runs after the cover autopsy, in the same runtime. It rebuilds the wrist mask without the 20% size cap, paints any pixels that still differ, places another object on the camera-to-bowl ray, rescores the feature rule on more task-0 frames, covers one frame of task 1, and rolls out full episodes with a 10-step re-query. Results go to `outputs/permanence/paper_claim/`.

## Remaining gaps

After the paper-claim cell, the remaining-gaps cell places the ramekin on the agent-camera ray and searches a second free body along the wrist-camera ray. When that body exists, both objects are placed in one scene. The cell also forwards a composite whose agent image comes from the agent-only render and whose wrist image comes from the wrist-only render, and it labels that composite as two renders. The closed loop passes each action chunk through the policy postprocessor before `env.step`, prints the raw and unnormalized mean absolute values, checks 40 steps of base actions against a zero action, then rolls out base, covered, and absent for 220 steps with a re-query every 10 actions. Results go to `outputs/permanence/remaining_gaps/`.

## Tight cover

After the remaining-gaps cell, the last cell forwards the smallest plate placement that still covers the bowl in both cameras, and it forwards the larger plate beside it. On the held frame it also tries a closer, smaller plate and records the wrist-camera distance when that plate fills the frame. The closed loop resets the episode and clears the done flag before every edit, then rolls out the visible bowl, the gray paint, the removed bowl, and the tight plate for 10 episodes. Results go to `outputs/permanence/tight_cover/`.

## Conclusion

After the tight-cover cell, the last permanence cell reuses those two frames and plate placements. It counts bowl pixels the plate left in place, paints them, and compares that action with full gray paint and with removal. It does not roll out again. If `scripts/tight_cover.py` is missing, the cell writes that module and leaves `outputs/permanence/tight_cover/` in place. Results go to `outputs/permanence/conclusion/`.

## Action-expert occlusion contrast

The action-expert cell in `final_smaj.ipynb` is a separate experiment. It clones `feature/akhidre-transcoder-circuit-tracing` into `/content/pi05-run` and leaves `/content/groot-run` and `outputs/permanence/` in place. It loads an existing `step_*.pt` checkpoint when one is already on disk or Drive, and otherwise trains once with `scripts/train_pi05_transcoders.py`. Feature discovery needs the LeRobot copy of `HuggingFaceVLA/libero`; the permanence cache does not include that dataset, so the cell reuses a local copy or Hub snapshot if one exists and otherwise downloads episodes `0,1,2` only, then links it to `HF_HOME/lerobot/HuggingFaceVLA/libero`. Feature discovery and the circuit tracer are the scripts already on that branch. The only new program is `occlusion_contrast.py`. It replays LIBERO demonstrations without letting Pi0.5 move the arm, scores one grasp feature on `held_visible`, `held_hidden`, and `gone`, and writes `/content/pi05-run/outputs/occlusion_contrast/verdict.json`. If that feature does not pass, the tracer is not run.

## Research zip

After the experiment cells, paste the zip cell from `final_smaj.ipynb` into the same runtime. It writes `/content/pi05_permanence_research.zip` and copies that archive to Drive when the shared folder is mounted.

Keep:

- the Colab notebook (`final_smaj.ipynb` or a download name such as `3final_smaj (1).ipynb`)
- experiment source and tests
- `outputs/permanence/*/summary.json` and the PNG contact sheets
- `outputs/occlusion_contrast/verdict.json` if that cell ran
- the layer-5 `transcoder.pt` under `controlled_contrasts/`

Skip:

- `lerobot-venv/`
- `hf_home/` and Drive `hf_home.tar`
- LIBERO `*.hdf5` archives
- Pi0.5 and action-expert `*.pt` weights (those stay on HuggingFace or Drive)
- `*.npz` replay dumps
- `secrets/` and `HF_TOKEN`

Locally:

```bash
python pack_research.py --zip pi05_permanence_research.zip
```

## Tests

The simulator and the policy are not available here. The mask, score, and transcoder checks run locally:

```bash
python -m unittest tests.test_controlled_contrasts tests.test_feature_count_sweep tests.test_token_position_patch tests.test_token_subspace tests.test_occlusion_features tests.test_paper_gaps tests.test_cover_autopsy tests.test_paper_claim tests.test_remaining_gaps tests.test_tight_cover tests.test_occlusion_contrast tests.test_conclusion tests.test_pack_research
```
