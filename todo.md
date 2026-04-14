[ ] LOOK AT THIS GUIDANCE https://gemini.google.com/app/3ba4cb6614c01063

Gemini’s concrete feedback, reduced to the parts that matter for this
  repo:

  1. The best path is not more dataset mining. It is a model-side
     instruction swap experiment.
     Keep image/state fixed, flip only the instruction at t0, and record
     predicted actions right after the switch.
  2. tail(A) + head(B) stitching is only defensible if state continuity is
     tightly controlled.
     Minimum constraints:
      - end-effector state continuity
      - near-identical visual layout
      - physically feasible new instruction
      - awareness that history/context windows can contaminate the result
  3. Gemini explicitly said the stronger experiment is the latent / model-
     side swap:
      - static or slow-moving observations
      - swap text embedding at t0
      - analyze the action step response in time and frequency domains
  4. Its ranked plan for this repo:
      - Phase 1: inference on a held-constant frame, flip instruction at
        step 5
      - Phase 2: search state-matched episode pairs and stitch only as a
        secondary experiment
      - Phase 3: run the spectral analysis on the resulting action traces
  5. Failure modes it flagged:
      - context-window pollution
      - action de-normalization / binning artifacts
      - impossible post-switch commands

  So the external feedback agrees with the direction I was already
  pushing: the real fix is to build the instruction-swap inference
  experiment, not keep pretending the public RLDS already contains t0.

  If you want, I’ll do the next concrete step now: implement experiments/
  instruction_step_inference.py plus a minimal mock/policy adapter path in
  models/vla_wrapper.py so this repo can run the proper switch experiment
  end-to-end.
