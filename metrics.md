Based on your PPO training code, here are all the metrics you're logging and how to monitor them:

## Core Performance Metrics (Primary Focus)

### **`train/mean_reward`** & **`eval/mean_reward`**
- **What**: Rolling average of episode rewards during training vs. evaluation performance
- **Monitor**: Primary success indicator - should trend upward toward your threshold (200 for LunarLander)
- **Action**: If plateauing, adjust learning rates or entropy coefficient

### **`train/total_steps`**
- **What**: Total environment steps consumed by the agent
- **Monitor**: Sample efficiency - fewer steps to reach threshold = better
- **Action**: Compare across hyperparameter settings to find most sample-efficient config

## Policy Learning Metrics

### **`epoch/policy_loss`**
- **What**: PPO clipped policy gradient loss
- **Monitor**: Should generally decrease but may fluctuate due to clipping
- **Action**: If consistently increasing, reduce learning rate or increase clip epsilon

### **`epoch/entropy`**
- **What**: Policy action distribution entropy (exploration measure)
- **Monitor**: Should start high and gradually decrease as policy becomes more deterministic
- **Action**: If drops too quickly, increase entropy coefficient; if too high, decrease it

### **`epoch/clip_fraction`**
- **What**: Fraction of policy updates that hit the PPO clipping bounds
- **Monitor**: Should be 0.1-0.3; higher means aggressive policy changes
- **Action**: If too high (>0.5), reduce learning rate or decrease clip epsilon

### **`epoch/kl_div`** & **`epoch/approx_kl`**
- **What**: KL divergence between old and new policies
- **Monitor**: Should be small (<0.1); large values indicate unstable training
- **Action**: If too high, reduce policy learning rate

## Value Function Metrics

### **`epoch/value_loss`**
- **What**: Mean squared error between predicted and actual returns
- **Monitor**: Should decrease over time as value function improves
- **Action**: If not decreasing, increase value learning rate or network capacity

### **`epoch/explained_var`**
- **What**: How well value function explains return variance (0-1 scale)
- **Monitor**: Should increase toward 1.0; >0.7 is good
- **Action**: If low, the value function isn't learning well - check value_lr or network size

### **`train/advantage_mean`** & **`train/advantage_std`**
- **What**: Statistics of computed advantages (GAE)
- **Monitor**: Mean should be near 0; std indicates advantage magnitude
- **Action**: If mean drifts from 0, value function may be biased

## Data Collection Metrics

### **`rollout/queue_updated`** vs **`rollout/queue_miss`**
- **What**: Success rate of async rollout collection
- **Monitor**: More updates than misses indicates healthy data pipeline
- **Action**: If many misses, increase queue size or reduce rollout interval

### **`rollout/steps_per_second`**
- **What**: Environment interaction speed
- **Monitor**: Higher is better for training efficiency
- **Action**: Use for comparing sync vs async modes

## Monitoring Strategy




## Red Flags to Watch For

1. **Reward plateau** with high clip_fraction → Reduce learning rates
2. **Value loss not decreasing** → Value function isn't learning properly
3. **Entropy drops to near 0** → Agent stopped exploring, increase entropy_coef
4. **Large KL divergence spikes** → Policy updates too aggressive
5. **Many rollout queue misses** → Data collection bottleneck

Focus primarily on `eval/mean_reward` trending toward your threshold, with `explained_var` and `entropy` as health checks for your learning process.