# State: Seed Difference, Round Number, Probability Difference Binned (10 bins)
# Action: 0 or 1 (Team 1 or Team 2)
# Reward: immediate reward of 10, 20, 40, 80, 160, or 320 based on the Round, 
#         and a delayed reward that is the bracket's total score at the end
# Use Epsilon-Greedy strategy to balance exploration and exploitation
# Use a Discount Factor of 0.9
# 1000 episodes, where each episode is a randomly-selected complete bracket

# Seed winning probability compared to predicted probability
# Percentage chance of making it to the next round

