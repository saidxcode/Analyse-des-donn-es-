# NumPy is imported, seed is set
import numpy as np 
# Initialize random_walk
random_walk=[0]

# Initialize dictionary to track heads and tails
dice_counts = {"heads": 0, "tails": 0}

# Complete the loop
for x in range(1000) :
    # Set step: last element in random_walk
    step=random_walk[-1]

    # Roll the dice
    dice = np.random.randint(1,7)

    # Track heads (dice <= 2) and tails (dice > 2)
    if dice <= 2:
        dice_counts["heads"] += 1
    else:
        dice_counts["tails"] += 1

    # Determine next step
    if dice <= 2:
        step = step - 1
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    # append next_step to random_walk
    random_walk.append(step)

# Print random_walk
print(random_walk)

# Print dice counts
print("\nDice Results:")
print(dice_counts)
import matplotlib.pyplot as plt

# Plot random_walk
plt.plot(random_walk)

# Show the plot
plt.show()