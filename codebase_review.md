Instant Policy: Comprehensive Analysis
Paper Overview
Title: Instant Policy: In-Context Imitation Learning via Graph Diffusion Authors: Vosylius and Johns Published: November 2024 (arXiv:2411.12633) Award: Best Paper at ICLR 2025 Robot Learning Workshop Website: https://www.robot-learning.uk/instant-policy
Core Concept
Instant Policy enables robots to learn new tasks instantly from just 1-2 demonstrations without further training. This is achieved through:
In-Context Imitation Learning (ICIL): The model adapts to new tasks at inference time by conditioning on demonstrations
Graph Diffusion: Models the task as a graph generation problem with learned diffusion processes
Pseudo-Demonstrations Training: Uses arbitrary simulated trajectories as infinite training data
Key Innovation
Unlike traditional imitation learning that requires task-specific training, Instant Policy learns a meta-policy that can instantly adapt to new tasks by observing demonstrations, similar to how large language models perform in-context learning.
Codebase Architecture
Directory Structure

instant_policy/ip/
├── configs/
│   └── base_config.py          # Central configuration
├── models/
│   ├── model.py                # AGI (main model)
│   ├── diffusion.py            # GraphDiffusion (Lightning wrapper)
│   ├── graph_rep.py            # Heterogeneous graph construction
│   ├── scene_encoder.py        # PointNet++ encoder
│   ├── graph_transformer.py    # Graph transformer blocks
│   └── occupancy_net.py        # Scene reconstruction
├── utils/
│   ├── common_utils.py         # SE(3) transforms, positional encoding
│   ├── data_proc.py            # Data preprocessing
│   ├── normalizer.py           # Action normalization
│   ├── rl_bench_utils.py       # RLBench integration
│   ├── rl_bench_tasks.py       # Task definitions (17 tasks)
│   └── running_dataset.py      # PyTorch dataset
├── train.py                    # Training script
├── eval.py                     # RLBench evaluation
├── deployment.py               # Real robot deployment
└── prepare_data.py             # Data preparation
Model Architecture: AGI
High-Level Components

AGI Model
├── SceneEncoder (PointNet++)
│   └── Pre-trained, frozen point cloud feature extraction
├── Graph Representation
│   ├── Scene nodes (16 per observation)
│   └── Gripper nodes (6 representing gripper geometry)
├── Three Graph Transformers
│   ├── Local Encoder (processes current observation)
│   ├── Conditional Encoder (aggregates demonstrations)
│   └── Action Encoder (generates action predictions)
└── Prediction Heads
    ├── Translation (3D)
    ├── Rotation (3D angle-axis)
    └── Gripper (1D open/close)
1. Scene Encoder (PointNet++)
File: scene_encoder.py
Architecture: Two-stage hierarchical point cloud encoder
SA Module 1:
Input: 3D points with 128-dim features
Downsampling: 12.5% (radius 0.125)
Output: 256-dim features
SA Module 2:
Input: 256+3 dims
Downsampling: 6.25% (radius 0.0625)
Output: 512-dim embeddings
Novel Feature: PointNetConvPE with Fourier positional encoding (10 frequencies)
Status: Pre-trained and frozen during policy training
2. Graph Representation
File: graph_rep.py Heterogeneous Graph Structure: Node Types:
Scene nodes: 16 sampled points from point clouds (per timestep)
Gripper nodes: 6 fixed points representing gripper geometry

Positions (relative to gripper frame):
[0, 0, 0]        # Center
[0, 0, -0.03]    # Back
[0, ±0.03, 0]    # Sides (2 points)
[0, ±0.03, 0.03] # Fingers (2 points)
Edge Types (7 relationship types):
rel: Spatial relations within current observation
scene → scene
scene → gripper
gripper → gripper
rel_demo: Spatial relations within demonstrations
rel_action: Spatial relations for action prediction
cond: Demonstration conditioning (demo gripper → current gripper)
demo: Temporal connections in demonstrations
time_action: Temporal connections in action sequence
rel_cond: Conditional spatial relations
Edge Attributes:
Positional encoding: 126-dim (2 × 63)
Encodes relative 3D positions with sinusoidal features
Formula: PE(pos) = [x, sin(f₁x), cos(f₁x), ..., sin(f₁₀x), cos(f₁₀x)] for each dimension
3. Graph Transformers
File: graph_transformer.py Architecture:
Multi-head attention (8 heads)
Hidden dimension: 1024
Layers: 2 (configurable)
Activation: GELU
Residual connections: x = attention(x) + mlp(x)
Three Specialized Instances:
Local Encoder: Processes current observation
Edges: scene-scene, scene-gripper, gripper-gripper
Aggregates local spatial context
Conditional Encoder: Aggregates demonstration information
Edges: cond, demo, rel_demo
Propagates demo context to current state
Action Encoder: Generates action predictions
Edges: time_action, rel_action, rel_cond
Temporal reasoning over prediction horizon
4. Diffusion-Based Action Generation
File: diffusion.py DDIM Scheduler Configuration:
Beta schedule: "squaredcos_cap_v2"
Training steps: 100
Inference steps: 8
Noise type: Gaussian
Action Representation:
SE(3) transforms: 4×4 matrices (rotation + translation)
Gripper state: Scalar (0=closed, 1=open)
Prediction horizon: 8 timesteps
Training Process:
Sample random diffusion timestep (0-99)
Add Gaussian noise to ground truth actions
Model predicts action corrections (deltas)
Compute L1 loss on predictions
Backpropagate with gradient clipping (norm=1)
Inference Process (Iterative Denoising):
Initialize random noisy actions
For each diffusion step (8 iterations):
Forward pass through graph transformers
Predict translation, rotation, gripper deltas
Apply DDIM scheduler step
Update actions via SE(3) composition
Return final denoised action sequence
Loss Functions:
Training: L1 loss on predicted deltas
Validation: SE(3) loss (translation L2 + rotation angle error)
Training Pipeline
Configuration
File: base_config.py Key Parameters:
Batch size: 16
Learning rate: 1e-5
Weight decay: 1e-2
Trajectory horizon: 10 (demo length)
Prediction horizon: 8 (action sequence length)
Number of demonstrations: 2
Scene nodes: 16 per observation
Action Space Bounds:
Translation: ±0.01m per step
Rotation: ±3° per step
Training Script
File: train.py PyTorch Lightning Setup:
Mixed precision (16-bit)
Gradient clipping: max norm 1.0
Validation every 20,000 steps
Checkpoint every 100,000 steps
Logging every 500 steps
Weights & Biases integration
Optimizer: AdamW
LR: 1e-5
Weight decay: 1e-2
Optional cosine scheduler with warmup (1000 steps)
Data Processing
File: prepare_data.py Pipeline:
Load RLBench demonstrations
Downsample point clouds to 2048 points
Remove statistical outliers
Extract trajectory waypoints (10 per demo)
Compute relative SE(3) transforms between waypoints
Pre-compute scene encoder embeddings (optional)
Save as PyTorch geometric Data objects
Data Format:

{
    'pcds': [np.array(N, 3)],      # Point clouds
    'T_w_es': [4×4 matrix],        # End-effector poses
    'grips': [float 0-1]           # Gripper states
}
Evaluation & Deployment
RLBench Evaluation
File: eval.py Supported Tasks (17 total):
lift_lid, phone_on_base, open_box, slide_block, close_box
basketball, buzz, close_microwave, plate_out
toilet_seat_down, toilet_seat_up, toilet_roll_off, open_microwave
lamp_on, umbrella_out, push_button, put_rubbish
Evaluation Protocol:
Collect N demonstrations from RLBench task
Run M rollouts (default 10)
For each rollout:
Reset environment
Execute policy with 8 diffusion steps
Track success/failure
Report success rate
Action Mode: MoveArmThenGripper with IK solver
End-effector pose control via inverse kinematics
Discrete gripper commands (open/close)
Real Robot Deployment
File: deployment.py Deployment Pipeline:
Load pre-trained model checkpoint
Collect 1-2 human demonstrations (point clouds + poses)
Pre-compute and cache demonstration embeddings
Control loop:
Capture current RGB-D observation
Segment point clouds
Transform to end-effector frame
Run inference (4-8 diffusion steps)
Execute predicted actions
Repeat until task completion
Requirements:
Segmented point clouds (objects separated)
Markovian tasks (no long-term memory needed)
Clean demonstrations
At least 2 objects in scene
Novel Technical Contributions
1. Graph-Based In-Context Learning
Heterogeneous graphs enable flexible conditioning on variable numbers of demonstrations
Different edge types separate spatial, temporal, and conditional information flow
Dynamic graph topology adapts to batch size and number of demos
2. SE(3)-Aware Action Prediction
Actions represented as 4×4 transformation matrices
Preserves group structure (composition via matrix multiplication)
Separate prediction heads for translation, rotation, gripper
3. Gripper Point Cloud Representation
6 points capture gripper geometry (not just end-effector pose)
Enables modeling of gripper orientation and contact points
Transformed in SE(3) during graph updates
4. Dual Positional Encoding
Scene encoder: Fourier features on local point neighborhoods
Graph edges: Sinusoidal encoding of relative 3D positions
Multi-scale: Different frequencies for different layers
5. Diffusion for Structured Prediction
DDIM scheduler enables iterative refinement
Coarse-to-fine action generation
Trade-off between speed (8 steps) and quality (100 steps)
6. Pre-trained Scene Encoder
PointNet++ trained separately and frozen
Reduces policy training complexity
Consistent geometry understanding across tasks
Key Insights from Codebase
Architectural Design Decisions
Why heterogeneous graphs?
Separate node types for scene and gripper allow specialized processing
Edge types encode different relationships (spatial, temporal, conditional)
Enables information flow from demos to current state to predicted actions
Why 6 gripper nodes instead of 1?
Captures gripper geometry (fingers, palm, orientation)
Important for contact-rich tasks
Enables better modeling of gripper-object interactions
Why diffusion for actions?
Handles multimodal action distributions
Iterative refinement improves prediction quality
Enables trade-off between speed and accuracy
Why freeze scene encoder?
Pre-trained on large point cloud datasets
Reduces training complexity
Prevents overfitting on limited demo data
Performance Optimizations
torch.compile(): Optional compilation for 2-3x inference speedup
Pre-computed embeddings: Cache demo scene encodings
Mixed precision: 16-bit training for speed
Batch processing: Vectorized operations across demos and timesteps
Training Insights
Pseudo-demonstrations: Can train on arbitrary trajectories (not just expert demos)
Data augmentation: Random gripper state inversion, rotations
Gradient clipping: Essential for stable training (norm=1)
Long training: 50 billion iterations (though checkpoints saved regularly)
Critical Files Reference
model.py - AGI model (239 lines)
diffusion.py - GraphDiffusion wrapper (243 lines)
graph_rep.py - Graph construction (360+ lines)
scene_encoder.py - PointNet++ (100 lines)
graph_transformer.py - Transformers (47 lines)
train.py - Training script
eval.py - RLBench evaluation
base_config.py - Configuration
Summary
Instant Policy is a sophisticated in-context imitation learning system that combines:
Graph neural networks for structured reasoning over demonstrations
Diffusion models for iterative action refinement
Point cloud processing for visual scene understanding
SE(3) geometry for robotic manipulation
The key innovation is the ability to instantly adapt to new tasks from 1-2 demonstrations without retraining, achieved through a graph-based representation that elegantly handles variable numbers of demonstrations and iterative diffusion-based action generation.
Sources
Instant Policy: In-Context Imitation Learning via Graph Diffusion (arXiv)
Instant Policy Project Website
OpenReview Forum
User approved the plan
Perfect! I've completed a comprehensive analysis of both the Instant Policy paper and the entire codebase. The analysis document has been created and is now available at C:\Users\srini\.claude\plans\zippy-enchanting-hippo.md.
Summary of What I Found
Paper Insights
Instant Policy enables robots to learn new manipulation tasks instantly from just 1-2 demonstrations
Won Best Paper Award at ICLR 2025 Robot Learning Workshop
Uses graph diffusion for in-context imitation learning
Codebase Deep Dive
Model: AGI (Advanced Graph Imitation) with sophisticated heterogeneous graph neural networks
Key Components:
Pre-trained PointNet++ scene encoder (frozen)
3 specialized graph transformers for local, conditional, and action encoding
DDIM diffusion with 8-step inference for real-time performance
6-point gripper representation capturing full geometry
SE(3)-aware action prediction with 4×4 transformation matrices
Architecture Highlights
7 different edge types for information flow (spatial, temporal, conditional)
Dual positional encoding (Fourier in scene encoder + sinusoidal in graph edges)
Iterative diffusion-based action refinement
PyTorch Lightning training with mixed precision
The analysis document contains detailed breakdowns of the architecture, training pipeline, data processing, evaluation protocols, and key design decisions with file references throughout the codebase. Sources:
Instant Policy: In-Context Imitation Learning via Graph Diffusion (arXiv)
Instant Policy Project Website
OpenReview Forum