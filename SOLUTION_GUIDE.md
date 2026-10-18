# CS 557 HW1: completed implementation and run guide

Source: `qtris123/CS557-HW1` at commit `ed0eff0a37c6ef2a836a236067f6753a259955cd`.
Read against all five pages of `CS_557_Fall_2026_Assignment1.pdf`.

This package supplies completed code and scripts that produce the required materials when you run the experiments. Short correctness checks are not evidence of convergence. Do not describe expected trends below as measured findings.

## 1. What was completed

| File | Implementation |
| --- | --- |
| `dqn.py` | Uniform replay, proportional PER, epsilon-greedy actions, DQN and Double DQN targets; terminal-batch and PER-loss fixes |
| `networks.py` | Dueling value/advantage aggregation for vector observations |
| `cnn_networks.py` | Dueling value/advantage aggregation for frame stacks |
| `utils.py` | Consistent CNN normalization during greedy evaluation; restores network mode |
| `main.py`, `logger.py` | Seed, run names, configuration recording, separate checkpoint directories, optional videos |
| `requirements_hw1.txt` | Dependencies for the two Box2D environments without unrelated MuJoCo/Atari dependencies |
| `verify_solution.py` | Numerical checks of targets, replay, PER weighting, dueling invariants, terminal handling, CNN evaluation |
| `run_experiments.py` | Prints or executes the 12-run plan; reuses completed runs and refuses conflicting logs |
| `export_plots.py` | Exports six comparison figures and an actual-results CSV |
| `package_submission.py` | Packages your finished PDF, completed code, and exactly two baseline DQN run logs |

Use these completed files directly, or compare `solution.patch` with the original starter. Do not combine the snippet-only changes with an otherwise uncorrected starter: the fixes in section 3 also matter.

## 2. Core solution snippets

### Uniform replay (`dqn.py`)

```python
def push(self, *args):
    self.memory.append(Transition(*args))

def sample(self, batch_size):
    return random.sample(self.memory, batch_size)
```

### Prioritized replay (`dqn.py`)

Use raw priorities `p_i = |delta_i| + eps`, sample with probability `P(i) = p_i**alpha / sum(p_j**alpha)`, and weight each sample by `(N * P(i))**(-beta)`. Sampling is with replacement. Keep priorities un-exponentiated when updating them.

```python
def sample(self, batch_size):
    if not self.memory:
        raise ValueError("Cannot sample an empty replay buffer")
    priorities = self.priorities[:len(self.memory)].astype(np.float64)
    probs = priorities ** self.alpha
    probs /= probs.sum()
    indices = np.random.choice(len(self.memory), batch_size, p=probs)
    samples = [self.memory[i] for i in indices]
    weights = (len(self.memory) * probs[indices]) ** (-self.beta)
    weights /= weights.max()
    return samples, indices, torch.tensor(weights, dtype=torch.float32)

def update_priorities(self, indices, td_errors):
    for i, error in zip(indices, np.asarray(td_errors).reshape(-1)):
        self.priorities[i] = abs(float(error)) + self.eps
```

`alpha=0` gives uniform probabilities; `beta=0` gives unit weights. The starter's batch-maximum normalization is preserved. Beta remains fixed at its configured value; no unrequested annealing schedule is added.

### Epsilon-greedy actions (`dqn.py`)

Retain the provided epsilon schedule and preprocessing. The two missing return statements are:

```python
# Exploitation branch, inside torch.no_grad():
return self.policy_net(state_tensor).argmax(dim=1).item()

# Exploration branch:
return self.action_space.sample()
```

Epsilon decays by environment steps, not by episodes. `--num_epochs` means training episodes.

### Complete `optimize_model` replacement (`dqn.py`)

This version is equivalent to the completed implementation, with the important shape and terminal-state handling shown together.

```python
def optimize_model(self):
    if len(self.memory) < self.batch_size:
        return
    if self.use_per:
        transitions, indices, weights = self.memory.sample(self.batch_size)
        weights = weights.to(self.device)
    else:
        transitions = self.memory.sample(self.batch_size)

    batch = Transition(*zip(*transitions))

    def stack_states(states):
        if self.is_cnn:
            encoded = [utils.preprocess_image(s) for s in states]
        else:
            encoded = [utils.encode_obs(s, self.env.observation_space)
                       for s in states]
        return torch.as_tensor(np.stack(encoded), dtype=torch.float32,
                               device=self.device)

    states = stack_states(batch.state)
    actions = torch.tensor(batch.action, dtype=torch.long,
                           device=self.device).unsqueeze(1)
    rewards = torch.cat(batch.reward).to(self.device, dtype=torch.float32)
    mask = torch.tensor([s is not None for s in batch.next_state],
                        dtype=torch.bool, device=self.device)
    q = self.policy_net(states).gather(1, actions).squeeze(1)  # [B]
    next_values = torch.zeros(self.batch_size, device=self.device)
    with torch.no_grad():
        if mask.any():
            next_states = stack_states([s for s in batch.next_state
                                        if s is not None])
            if self.use_double_dqn:
                best = self.policy_net(next_states).argmax(1, keepdim=True)
                next_values[mask] = self.target_net(next_states).gather(1, best).squeeze(1)
            else:
                next_values[mask] = self.target_net(next_states).max(1).values
        targets = rewards + self.gamma * next_values
        td_errors = targets - q

    losses = nn.functional.smooth_l1_loss(q, targets, reduction="none")
    loss = (weights * losses).mean() if self.use_per else losses.mean()
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
    self.optimizer.step()
    if self.use_per:
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
    return loss.item()
```

DQN uses `r + gamma * max_a Q_target(s_next, a)`. Double DQN chooses `argmax_a Q_online(s_next, a)` and evaluates that action using `Q_target`. Both use just the immediate reward on a true terminal transition. Gradients flow through the current online Q-values, not the target computation.

### Dueling networks

`networks.py`, `DuelingMLPNetwork.forward`:

```python
def forward(self, x):
    features = self.feature_layer(x)
    value = self.value_head(features)
    advantages = self.advantage_head(features)
    return value + advantages - advantages.mean(dim=1, keepdim=True)
```

`cnn_networks.py`, `DuelingCNNNetwork.forward`:

```python
def forward(self, x):
    features = self.features(x).flatten(1)
    value = self.value_stream(features)
    advantages = self.advantage_stream(features)
    return value + advantages - advantages.mean(dim=1, keepdim=True)
```

Subtract the mean across actions separately for each state. Do not average across the batch.

## 3. Starter bugs corrected beyond the TODOs

- **PER loss broadcasting:** original losses are `[B, 1]` and weights are `[B]`; direct multiplication yields `[B, B]`. The completed implementation squeezes losses to `[B]` before multiplication.
- **All-terminal minibatches:** do not stack the next-state list or run the next-state networks when that list is empty.
- **CNN evaluation:** the CarRacing wrapper returns uint8 `[12, 84, 84]` stacks. Training divides by 255, and evaluation must do the same. Use `torch.from_numpy(preprocess_image(obs)).float().unsqueeze(0).to(device)`.
- **CNN flattening:** use `features.flatten(1)` in both CNN forward methods. `.view(batch_size, -1)` can crash on noncontiguous/channel-last convolution outputs produced by stacked observations.
- **Termination versus truncation:** replace `if terminated or truncated: next_state = None` with `if terminated: next_state = None`. End the episode on either flag, but bootstrap at an external time limit using the final observation. These environments are not auto-reset wrappers.
- **Target updates:** schedule updates using `self.steps_done % self.target_update`, not the episode-local `t`. The configured interval now refers consistently to environment steps. For `tau > 0`, apply the soft update; the starter convention `tau == 0` requests hard copies.
- **Hidden size:** pass the CLI setting into network constructors; the starter silently ignored it. The plan uses 128 for both MLP hidden layers and the CNN fully connected hidden size.
- **Run identity and evaluation:** log all hyperparameters, seed initialization/environment/action sampling, use separate checkpoint paths, evaluate at the final episode, and restore the previous network training mode after evaluation.
- **Video memory:** optional rollout clips capture at most 300 steps and resize RGB frames to at most 320 pixels on the longest side. Full greedy evaluation still uses complete episodes. A completion marker is written only after logging closes, so a process killed during video capture cannot be reused as a completed run.

Apply the same fixes and code version to every experiment, including baselines.

## 4. Experiment matrix

The PDF's run-count wording is inconsistent. This plan covers the stricter interpretation: both Part 1 choices on each environment, two new agent/replay choices per environment, and two separate three-value hyperparameter sweeps. No need to run all six agent/replay combinations.

| Part | Environment | Run names | New runs |
| --- | --- | --- | ---: |
| 1 | Both | `p1_dqn`, `p1_double_per` | 4 |
| 2A | Both | `p2a_double`, `p2a_dueling_per` | 4 |
| 2B, learning rate | LunarLander | `p2b_lr_1em5`, existing `p1_double_per`, `p2b_lr_1em3` | 2 |
| 2B, epsilon decay | LunarLander | `p2b_decay_5000`, existing `p1_double_per`, `p2b_decay_50000` | 2 |
| Total | | | **12** |

Both sweeps use **Double DQN + PER on LunarLander**. Change only learning rate in the first sweep (`1e-5, 1e-4, 1e-3`) and only epsilon decay in the second (`5000, 15000, 50000`). The baseline is the identical middle point in both sweeps, so reuse its logs rather than repeating it.

Fixed values: batch size 64, hidden size 128, replay capacity 10000, epsilon start/end 0.9/0.05, gamma 0.99, target update every 2 environment steps, tau 0.005, PER alpha/beta 0.6/0.4, save/evaluate every 20 episodes, seed 0. Use **500 LunarLander episodes** and **100 CarRacing episodes**, as the handout suggests. Videos are optional and off in the experiment launcher to keep logs small.

Each evaluation uses greedy actions and the mean return over the same three evaluation seeds (`100000 + training_seed` through `100002 + training_seed`). They are separate from training reset seeds. One training seed is a compact assignment experiment, not a statistical robustness study.

## 5. Step-by-step execution

Run from the extracted solution directory. These are Linux/WSL terminal commands. Do not install the original requirements file in addition: it includes unrelated dependencies and obsolete pins.

1. Create and activate an environment, then install dependencies:

   ```bash
   conda create -n cs557rl python=3.10 -y
   conda activate cs557rl
   python -m pip install swig==4.3.1
   python -m pip install -r requirements_hw1.txt
   ```

   Box2D compiles a C++ extension. On Ubuntu/WSL, install `build-essential` if a compiler is missing (`sudo apt-get install build-essential`). Activate the environment before installation so `swig` is on PATH. If Python's compiler configuration names a missing `clang++`, install with `CC=gcc CXX=g++ python -m pip install -r requirements_hw1.txt` instead. Tested dependencies include PyTorch 2.5.1+cpu and Gymnasium 0.29.0 on Linux/Python 3.12. On an existing GPU environment, keep a PyTorch build compatible with that GPU.

2. Run the correctness checks, then optional short integration runs in a separate output directory:

   ```bash
   python verify_solution.py
   python main.py --env LunarLander-v2 --num_epochs 2 --save_interval 1 --batch_size 8 --run_name smoke --output_root smoke --no_video
   python main.py --env CarRacing-v2 --num_epochs 1 --save_interval 1 --batch_size 8 --run_name smoke --output_root smoke --use_double_dqn --use_dueling_dqn --use_per --no_video
   ```

   These short runs check execution only. Do not use their curves as the assignment experiments. On a machine without a display, `SDL_VIDEODRIVER=dummy` can be set before commands that record rollout videos.

3. Inspect and run the full experiment plan:

   ```bash
   python run_experiments.py
   python run_experiments.py --execute
   ```

   Or execute sequential parts separately:

   ```bash
   python run_experiments.py --part p1 --execute
   python run_experiments.py --part p2a --execute
   python run_experiments.py --part p2b --execute
   ```

   Run `p1` before exporting `p2b`: its Lunar Double-DQN+PER run is the shared baseline. Add `--videos` to the launcher if you want TensorBoard rollout videos. Completed matching runs are reused. Interrupted runs are not resumed: move their incomplete run directory aside before rerunning; do not mix event files from different attempts.

4. Generate the figures and inspect training:

   ```bash
   python export_plots.py
   tensorboard --logdir runs
   ```

   Open `http://localhost:6006`. For a remote machine, forward port 6006 to your local machine first. TensorBoard tags are `train-dqn/avg_loss`, `train-dqn/episode_reward`, `train-dqn/eval_mean_reward`, and `train-dqn/epsilon`.

5. Write your report using the six exported figures and measured values from `plots/summary.csv`. Export it to PDF, for example `Tri_Vo_hw1_report.pdf`. Include your full name inside it.

6. Build the submission archive, replacing the name arguments as needed:

   ```bash
   python package_submission.py --report Tri_Vo_hw1_report.pdf --firstname Tri --lastname Vo
   ```

   Upload `tri_vo_hw1.zip` to Brightspace. The script requires a real, completed report PDF and refuses to overwrite an existing submission archive. It packages only the two standard-DQN log directories as `selected_runs/LunarLander-v2/` and `selected_runs/CarRacing-v2/`, along with the completed code. It excludes other runs, checkpoints, environments, and intermediate plots.

## 6. Expected output materials

| Output | Purpose |
| --- | --- |
| `runs/lunarlander_v2/<run>_seed0/` | LunarLander TensorBoard events and `config.json` |
| `runs/carracing_v2/<run>_seed0/` | CarRacing TensorBoard events and `config.json` |
| `checkpoints/<environment_tag>/<run>_seed0/episode_N.pth` | Network checkpoints; not required in submission ZIP |
| `plots/part1_LunarLander-v2.png` | Part 1 baseline versus Double DQN + PER |
| `plots/part1_CarRacing-v2.png` | Part 1 baseline versus Double DQN + PER |
| `plots/part2A_LunarLander-v2.png` | Four agent/replay curves |
| `plots/part2A_CarRacing-v2.png` | Four agent/replay curves |
| `plots/part2B_lr.png` | Three learning rates, fixed agent/environment |
| `plots/part2B_epsilon_decay.png` | Three exploration decay values, fixed agent/environment |
| `plots/summary.csv` | Measured final/best evaluation return and mean training return over the last 10 episodes |
| `firstname_lastname_hw1.zip` | Final submission after you supply the completed PDF |

Each PNG contains training loss, training return, and evaluation return panels. Training curves use a trailing 10-observation mean and evaluation curves are unsmoothed; captions identify this. `python export_plots.py --smooth 1` exports raw curves instead.

## 7. What to explain in the report

The handout calls the training metric "accuracy," but this value-based RL implementation has no classification accuracy. State that you report **training episode return, mean Huber loss, and greedy evaluation return**. Do not invent an accuracy metric. A falling TD loss alone does not establish a better policy, especially across replay methods with different sampling distributions and weights.

- **Part 1:** explain replay decorrelation/reuse and target-network stabilization. Include both environments' standard DQN results and the Double-DQN+PER results.
- **Part 2A:** show all four agent/replay curves on each environment. Compare DQN versus Double DQN with uniform replay to assess the target rule; compare Double DQN with/without PER to assess replay. Dueling+PER versus Double-DQN+PER changes both architecture and target rule, so describe it as an agent comparison without attributing the difference solely to dueling.
- **Part 2B:** give the fixed settings and the exact three values in each sweep. Report final evaluation return, stability/oscillation, and how quickly useful returns appear. Discuss whether a larger learning rate caused instability or a smaller one learned more slowly only if your curves support it. Faster epsilon decay reduces exploration earlier; slower decay maintains it longer. Either can help or hurt depending on what was learned.
- State the training episode budgets, environment versions, training seed, evaluation protocol, and smoothing. Compare learning curves, not just the highest noisy checkpoint. Alpha controls how strongly replay prioritizes TD errors; beta controls importance correction. They stay fixed in this plan.

Keep analysis tied to your observed curves. If 100 CarRacing episodes do not learn a useful policy, report that outcome and possible causes rather than claiming success. The handout suggests budgets, not a guaranteed solution threshold.

The handout title says Fall 2026 while its due-date line says October 22, 2025. Use Brightspace for the authoritative deadline.
