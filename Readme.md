# Carla Lanekeeping

An overview of final project for 16299, Spring 2026 <br />
By Alex Xu and Vicky Zhu <br/>

## Final Project Report

### Video
[Video of Best Model](https://drive.google.com/file/d/1pCRysw866VCSPfoA9nWHS7UPhLVKljdw/view?usp=sharing)

### Abstract

This project develops and evaluates an autonomous driving agent specifically optimized for high-precision lane-keeping tasks.  Utilizing the CARLA simulation environment, a custom Gymnasium-compatible wrapper was engineered to facilitate Reinforcement Learning (RL) training.  The agent employs the Proximal Policy Optimization (PPO) algorithm to map a continuous action space (steering) from a simplified observation space consisting of lateral displacement and heading error.

To enhance driving stability, the system implements a look-ahead waypoint mechanism, allowing the agent to anticipate road curvature by calculating errors against future waypoints.  The reward function is strategically shaped to balance three competing objectives: maintaining high velocity, minimizing lateral deviation from the lane center, and ensuring steering smoothness to prevent oscillations.  Experimental results, derived from multi-episode evaluations in different maps, quantify performance through metrics including success rate, mean/max lateral error, and survival steps, providing a robust benchmark for RL-based lateral control stability.

### Introduction
The development of autonomous vehicles (AVs) represents one of the most significant shifts in modern transportation, promising to redefine safety, efficiency, and accessibility.  At the core of this transition is the ability of a vehicle to perceive its environment and make precise control decisions without human intervention.  While high-level navigation involves complex path planning and obstacle avoidance, the most fundamental requirement for any autonomous agent is lateral control, specifically lane keeping.

Lane keeping is the process of maintaining a vehicle's position within the boundaries of a travel lane.  It serves as the primary building block for Advanced Driver Assistance Systems (ADAS) and is the prerequisite for more complex maneuvers, such as lane changes, merging, and high-speed highway cruising.

### Motivation

The motivation for perfecting lane-keeping algorithms is rooted in two critical areas:

1. Enhancing Road Safety
Human error remains the leading cause of traffic accidents worldwide.  According to data from the National Highway Traffic Safety Administration (NHTSA), a significant percentage of highway fatalities are the result of unintended lane departures caused by driver fatigue, distraction, or medical emergencies.  Automated lane-keeping systems act as a "digital safeguard," providing continuous monitoring and corrective steering to prevent "run-off-road" collisions.

2. Foundational Control Complexity
From a robotics perspective, lane keeping is not a trivial task. It requires a tight coupling between perception (identifying lane markers) and actuation (adjusting steering angles).

The Challenge: Vehicles are non-linear dynamical systems operating in varying conditions (different speeds, road curvatures, and weather). Traditional control methods, like PID or Model Predictive Control (MPC), often require extensive tuning and precise mathematical models of the vehicle.

The RL Advantage: Using Proximal Policy Optimization (PPO) allows the agent to learn an optimal steering policy through trial and error in high-fidelity simulators like CARLA, enabling the system to adapt to complex road geometries without manual model linearization.

### Data Description

1. Success Rate
The **Success Rate** is the primary mission-completion metric.  It represents the percentage of test episodes where the vehicle successfully navigated the entire waypoint path without triggering a termination condition.
*   **Calculation:** `(Total Successful Episodes / Total Test Episodes) * 100`
*   **Success Condition:** Reaching the final waypoint in the `waypoint_list`.
*   **Failure Conditions:** Vehicle exiting the lane boundaries or becoming stationary for an extended period.

2. Lateral Error
Lateral error measures the perpendicular distance between the vehicle's center of gravity and the target trajectory (the centerline of the lane).
*   **Mean Lateral Error:** The average displacement over the duration of an episode.  This indicates the agent's overall precision and smoothness in tracking the lane center.
*   **Max Lateral Error:** The peak displacement recorded. This is a critical safety metric; a high max error indicates "swerving" or instability, even if the mean error remains low.

3. Survival Steps
Survival steps quantify the robustness and longevity of the agent during a single episode.
*   **Definition:** The total number of simulation ticks (frames) elapsed before the episode ends.
*   **Utility:** This is particularly useful during the early stages of training to track how long the model can "stay on the road" before failing, regardless of whether it reaches the final destination.

### Carla Setup

### Model Training

To facilitate efficient training, the environment supports both high-fidelity visual feedback and high-throughput headless modes.

Training with Visual Feedback
Recommended for initial debugging and observing agent behavior in real-time. Configure Environment: 
* In steer_RL.py, set settings.no_rendering_mode = False
* Launch CARLA: Execute ./CarlaUE4.sh -nosound
* Initiate Training: Run python3 train.py.

Headless Training (High Performance)
Recommended for long-running training sessions (e.g., 100k+ timesteps). Configure Environment: 
* In steer_RL.py, set settings.no_rendering_mode = True
* Launch CARLA: Execute ./CarlaUE4.sh -nosound -RenderOffScreen -quality-level=low
* Initiate Training: Run python3 train.py. 
* Use TensorBoard (tensorboard --logdir ./ppo_carlalog) to monitor convergence.

Hyperparameter Tuning Guide
The performance of the lane keeping agent is highly sensitive to the interaction between the environment's resolution ($d$) and the RL agent's foresight (look-ahead).
1. Waypoint Resolution ($d$)
In steer_RL.py, d = 0.5 defines the distance between waypoints in meters.
Trade-off: 
* A smaller $d$ provides a high-resolution path for the agent but increases the computational overhead and can lead to "noisy" reward signals. 
* A larger $d$ simplifies the path but may cause the agent to "miss" the apex of sharp curves.Tuning: 
If the car "oscillates" on straight lines, try increasing $d$ slightly to smooth out the target trajectory.

2. Throttle Management
In steer_RL.py, the current throttle is fixed at 0.4.  As speed increases, the lateral forces required to stay in the lane increase quadratically.  If the agent masters the track at 0.4 but fails at 0.6, it likely needs a more aggressive smoothness penalty to prevent over-steering at high speeds.

3. Look-ahead Index
In steer_RL.py, the look_ahead = 20 parameter determines which future waypoint is used to calculate steer_error.  With $d=0.5$ and $look\_ahead=20$, the agent is looking 10 meters into the future.  
* Short Look-ahead: Leads to "reactive" driving. The car stays centered on straights but reacts too late to curves.
* Long Look-ahead: Leads to "anticipatory" driving. The car starts turning early, which might cause it to "cut" the inside of a corner.

4. Reward Function Shaping
The reward function is the most critical component for balancing speed and safety.
In steer_RL.py, the current implementation follows this structure: 
$$R = w_1 \cdot \text{speed} - (w_2 \cdot |e_{lat}| + w_3 \cdot |e_{steer}| + w_4 \cdot |\Delta \text{steer}|)$$
* Speed Reward ($w_1$): Prevents the agent from simply standing still to avoid penalties.
* Lateral Penalty ($w_2$): The primary constraint. High values here make the car "stiff" and centered.
* Steer Error Penalty ($w_3$): Encourages the car to align its heading with the road's future direction.
* Smoothness Penalty ($w_4$): Crucial for preventing "jitter." If the car's steering wheel is shaking rapidly, increase this weight to penalize high $\Delta \text{steer}$ values.

Algorithm Selection
Proximal Policy Optimization (PPO) was chosen as the core learning algorithm because it provides a robust balance between performance and stability, which is critical for continuous control tasks like steering.  Unlike older Reinforcement Learning methods that can be hypersensitive to hyperparameter changes—often leading to "catastrophic forgetting" where the agent suddenly loses all driving ability—PPO utilizes a clipped surrogate objective.  This mechanism ensures that policy updates stay within a safe range, preventing the model from making erratic steering adjustments during the learning process. Furthermore, PPO’s Actor-Critic architecture allows the agent to effectively map continuous steering angles while simultaneously learning to predict the long-term value of its position within the lane.  This makes it particularly well-suited for the CARLA environment, where the agent must navigate high-dimensional physics with a high degree of precision.

### Model Evaluation

Once training is complete, the model is transitioned from a probabilistic learning phase to a deterministic evaluation phase.  This ensures that the agent always chooses the "best" known action rather than exploring random steering inputs.

Evaluation Setup
Episodes: 10 independent runs (set in evaluate.py)
Map: Town03 (set in steer_RL.py: client.load_world('Town03'))
Configuration: deterministic=True to eliminate stochastic exploration noise.
Termination Criteria: 
Episodes end upon a lane departure (Lateral Error > Width/2) or reaching the final destination.

Evaluation with Visual Feedback
Recommended for qualitative analysis—observing how the car handles specific turns and curves.
Environment Setup: 
* In laneEnv.py (or your environment script), set settings.no_rendering_mode = False.
* Launch CARLA: Execute ./CarlaUE4.sh -nosound.
* Run Evaluation: Execute python3 evaluate.py.
Note: You can uncomment time.sleep(0.05) in your evaluation script to watch the driving in real-time.

Headless Evaluation (Metric Collection)
Recommended for gathering quantitative data across many episodes quickly.
* Environment Setup: In laneEnv.py, set settings.no_rendering_mode = True.
* Launch CARLA: Execute ./CarlaUE4.sh -nosound -RenderOffScreen -quality-level=low.
* Run Evaluation: Execute python3 evaluate.py.
The script will output the final driving metrics to the terminal once all episodes are complete.

Quantitative Driving Metrics
We evaluate the agent across four key dimensions to gauge both safety and mission success:
* **Success Rate (%)**: Measures the reliability of the agent in completing the full designated route. This is the ultimate "pass/fail" test for the model.
* **Average Survival Time (Steps)**: Quantifies the robustness of the control policy. Higher survival steps indicate that even if the car fails, it can maintain lane stability for an extended duration.
* **Average Mean Lateral Error (m)**: Represents the agent's Precision. A low mean error shows the agent’s ability to consistently "hug" the lane centerline.
* **Average Max Lateral Error (m)**: Represents the agent's Stability. This captures the "worst-case" swerve. It is crucial for ensuring the vehicle never gets dangerously close to the lane boundary.

### Conclusions / Results

Training Performance
The PPO agent was trained for 100,000 timesteps in the CARLA Town03 environment. As shown in the TensorBoard logs, the ep_rew_mean (mean episodic reward) exhibited a classic reinforcement learning curve (graph in results/model_training_process.png):
* Initial Phase (0k–20k steps): The agent struggled to maintain lane position, frequently incurring heavy penalties from the -50.0 terminal crash penalty, resulting in mean rewards below -60.
* Learning Phase (20k–70k steps): A sharp upward trend occurred as the agent discovered the relationship between the look_ahead waypoint and steering actuation.
* Convergence (70k–100k steps): The reward stabilized in positive territory, peaking at a smoothed value of approximately 11.86. This indicates the agent learned to balance the speed_reward against the penalties for lateral deviation and steering jitter.

Quantitative Evaluation
The best performing model (lane_keeping_ppo4) was evaluated over 10 deterministic episodes. The agent achieved high precision and total reliability:
* Success Rate: 100%
* Avg Mean Lateral Error: 0.033 meters
* Avg Max Lateral Error: 0.114 meters
* Avg Survival Time: 250.0 steps
The results demonstrate exceptional tracking accuracy; an average mean lateral error of just 3.3 cm suggests the vehicle remains almost perfectly centered.  Even the "worst-case" deviation (11.4 cm) remains well within the safety margins of a standard lane width.

This project successfully implemented a robust lateral control system for autonomous lane keeping using Proximal Policy Optimization (PPO) within the CARLA simulator. By carefully shaping the reward function to penalize lateral error and steering instability while incentivizing speed, we developed an agent capable of navigating the complex geometries of Town03, and other maps, with a 100% success rate.  The integration of a look-ahead waypoint mechanism proved vital for stability, allowing the agent to anticipate curves rather than reacting to immediate errors.  The evaluation metrics confirm that the RL-based approach can achieve sub-decimeter precision, rivaling traditional control methods like PID or Pure Pursuit, while maintaining the flexibility to learn directly from simulated physics.

### Reflection: Iterative Model Optimization
The majority of the project lifecycle was dedicated to the iterative process of model training and hyperparameter fine-tuning.  This process revealed the delicate balance required between vehicle velocity and lateral stability.

Phase 1: The Baseline Model
Our initial approach utilized a conservative configuration with a simplified reward function:

$$R = \text{speed reward} - 0.5 \cdot e_{steer} - 0.5 \cdot e_{lat}$$

With a throttle of 0.3 and a sparse waypoint resolution ($d$) of 2.0 meters, the agent was able to navigate the track but did so with significant latency and low average speed.  While functional, this "slow-and-steady" policy lacked the precision needed for more complex maneuvers.  [Video of Initial Model Performance](https://drive.google.com/file/d/1pQK_nwMJJxryJVt2JcBxAPIl0f4A6eM_/view?usp=sharing)

Phase 2: Addressing Instability
As we attempted to increase the vehicle's speed, the agent began to exhibit high-frequency oscillations (the "jitter" effect).  The initial reward function failed to penalize aggressive steering changes, causing the car to over-correct and eventually lose lane stability.  We did tune the parameters of the reward function, however, it still wasn't enough.  [Video of One Bad Performance](https://drive.google.com/file/d/1jlyO6cjFCwlWOGzVbAwJ2zzVdzKKDGE5/view?usp=sharing)

Phase 3: The Optimized Configuration
To reach our best-performing model, we implemented three critical changes: 
* Smoothness Penalty: We introduced a steer_delta penalty (-0.2) to discourage rapid steering fluctuations, resulting in a significantly more stable driving line.
* Heading Prioritization: We doubled the weight of the steer_error penalty (to -1.0). This forced the agent to align more aggressively with future waypoints, allowing it to anticipate curves rather than reacting to them.
* Path Resolution: We reduced the waypoint distance ($d$) to 0.5 meters. This higher-resolution path provided the agent with more frequent feedback, which was essential for maintaining a low mean lateral error of 0.033m at higher velocities.

Final reward function: 

$$R = \text{speed reward} - (1.0 \cdot e_{steer} + 0.5 \cdot e_{lat} + 0.2 \cdot e_{\text{steer delta}})$$

### References
[1] Chen, Z., & Huang, X. (2017). End-to-End Learning for Lane Keeping of Self-Driving Cars. 2017 IEEE Intelligent Vehicles Symposium (IV), 1856-1860. https://users.wpi.edu/~xhuang/pubs/2017_chen_iv.pdf
[2] Mancuso, A. (2018). Study and implementation of lane detection and lane keeping for autonomous driving vehicles [Master's thesis, Politecnico di Torino]. WebThesis. https://webthesis.biblio.polito.it/9514/1/tesi.pdf
