# Bimanual Instant Policy Diagrams (Paper-Style)

This file provides presentation-ready diagrams for the bimanual extension using the same high-level style as the single-arm Instant Policy paper (data adapter -> graph encoder -> diffusion prediction).

## 1) Model Architecture (Bimanual, Relative/Local)

```mermaid
flowchart LR
  %% Inputs
  subgraph I[World-Frame Inputs]
    PW[Scene points P^W\n[B, N, 3]]
    TLW[T_W_L current\n[B,4,4]]
    TRW[T_W_R current\n[B,4,4]]
    GL[grip_L current]
    GR[grip_R current]
  end

  %% Adapter
  A[World -> Relative Adapter\n(build_obs_targets)]

  %% Relative observation
  subgraph O[Relative Observation]
    PL[P^L = (T_W_L)^-1 P^W\n[B,N,3]]
    PR[P^R = (T_W_R)^-1 P^W\n[B,N,3]]
    TLR[T_L_R = (T_W_L)^-1 T_W_R\n[B,4,4]]
    GLR[grip_L, grip_R]
  end

  %% Graph rep
  subgraph G[BimanualGraphRep]
    SL[scene_left nodes]
    SR[scene_right nodes]
    GLN[gripper_left nodes\n(keypoints + grip)]
    GRN[gripper_right nodes\n(keypoints + grip)]

    E1[local edges\nscene-scene\nscene-gripper\ngripper-gripper]
    E2[cross-arm edges\ngripper_L <-> gripper_R\nattrs from T_L_R / T_R_L]
  end

  ENC[GraphTransformer Encoder]

  %% Diffusion-conditioned prediction
  subgraph D[Diffusion-Conditioned Heads]
    DT[Sinusoidal time emb + node emb]
    HL[Left heads\nΔtrans_L, Δrot_L, Δgrip_L\n[B,P,G,7]]
    HR[Right heads\nΔtrans_R, Δrot_R, Δgrip_R\n[B,P,G,7]]
  end

  PW --> A
  TLW --> A
  TRW --> A
  GL --> A
  GR --> A

  A --> PL
  A --> PR
  A --> TLR
  A --> GLR

  PL --> SL
  PR --> SR
  TLR --> E2
  GLR --> GLN
  GLR --> GRN
  SL --> E1
  SR --> E1
  GLN --> E1
  GRN --> E1

  E1 --> ENC
  E2 --> ENC
  SL --> ENC
  SR --> ENC
  GLN --> ENC
  GRN --> ENC

  ENC --> DT
  DT --> HL
  DT --> HR
```

## 2) Invariance Contract (First Principles)

```mermaid
flowchart TB
  G[Global relabel g in SE(3)]

  subgraph W0[Original World Quantities]
    PW0[P^W]
    TL0[T_W_L]
    TR0[T_W_R]
    TLF0[T_W_L^future]
    TRF0[T_W_R^future]
  end

  subgraph W1[Relabeled World Quantities]
    PW1[P'^W = g P^W]
    TL1[T'_W_L = g T_W_L]
    TR1[T'_W_R = g T_W_R]
    TLF1[T'_W_L^future = g T_W_L^future]
    TRF1[T'_W_R^future = g T_W_R^future]
  end

  subgraph INV[Invariant Relative Terms Used by Model]
    PLI[P^L = (T_W_L)^-1 P^W]
    PRI[P^R = (T_W_R)^-1 P^W]
    TLI[T_L_R = (T_W_L)^-1 T_W_R]
    DLI[ΔT_L = (T_W_L)^-1 T_W_L^future]
    DRI[ΔT_R = (T_W_R)^-1 T_W_R^future]
  end

  subgraph INV2[After Relabeling]
    PLI2[P'^L = (T'_W_L)^-1 P'^W = P^L]
    PRI2[P'^R = (T'_W_R)^-1 P'^W = P^R]
    TLI2[T'_L_R = (T'_W_L)^-1 T'_W_R = T_L_R]
    DLI2[ΔT'_L = (T'_W_L)^-1 T'_W_L^future = ΔT_L]
    DRI2[ΔT'_R = (T'_W_R)^-1 T'_W_R^future = ΔT_R]
  end

  G --> W1
  W0 --> INV
  W1 --> INV2
```

## Notes for Slide Narration

- Key point: all high-bandwidth geometric signals are arm-local/relative.
- Cross-arm coupling is explicit through `T_L_R` edge attributes, not implicit world shortcuts.
- World-frame relabeling does not change the physically relevant state seen by the model.
