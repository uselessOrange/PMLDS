import numpy as np
import matplotlib.pyplot as plt

# Define the number of permutations and iterations
permutations = 3
num_iterations = 10
rmse_matrix=np.zeros([3,num_iterations,permutations])




losses_per_permutation=[]
val_losses_per_permutation=[]
# Loop over permutations
for permutation in range(permutations):
    losses=[]
    val_losses=[]
    
    # Loop over iterations
    for j in range(num_iterations):
        size = np.random.randint(5, 15)
        loss = np.random.normal(1, 2, size=size)
        losses.append(loss)
        val_losses.append(-loss)  # Append losses for each iteration to the main list

        rmse_matrix[0,j,permutation]=np.random.normal(1, 1000)
        rmse_matrix[1,j,permutation]=np.random.normal(1, 1000)
        rmse_matrix[2,j,permutation]=np.random.normal(1, 1000)
    losses_per_permutation.append(losses)
    val_losses_per_permutation.append(val_losses)





'''

# Assuming you have the 'losses_per_permutation' data

# Plotting all iterations for each permutation on the same plot
for i, permutation_losses in enumerate(losses_per_permutation):
    plt.figure(figsize=(8, 6))
    for loss in permutation_losses:
        plt.plot(np.arange(len(loss)), loss, label=f'Iteration')

    plt.xlabel('Loss Index')
    plt.ylabel('Loss Value')
    plt.title(f'Permutation {i+1} - All Iterations')
    plt.legend()
plt.show()

plt.boxplot(rmse_matrix[:][:][1])
plt.show()'''


'''


# Plotting all iterations for each permutation on the same plot
for i, permutation_losses in enumerate(losses_per_permutation):
    plt.figure(figsize=(8, 6))
    count = 0
    for loss in permutation_losses:
        temp1=loss
        temp2=val_losses_per_permutation[i][count]
        plt.plot(np.arange(len(loss)), loss, 'r', label=f'Iteration {count}')
        plt.plot(np.arange(len(val_losses_per_permutation[i][count])), val_losses_per_permutation[i][count], 'b', label=f'Iteration {count}')
        count += 1
    plt.xlabel('Loss Index')
    plt.ylabel('Loss Value')
    plt.title(f'Permutation {i+1} - All Iterations')
    plt.legend()
    plt.show()  # Show each plot for each permutation
'''


fig, axes = plt.subplots(len(losses_per_permutation), 1, figsize=(8, 6 * len(losses_per_permutation)))

# Plotting all iterations for each permutation as subplots
for i, permutation_losses in enumerate(losses_per_permutation):
    count = 0
    for loss in permutation_losses:
        axes[i].plot(np.arange(len(loss)), loss, 'r', label=f'Iteration {count}')
        axes[i].plot(np.arange(len(val_losses_per_permutation[i][count])), val_losses_per_permutation[i][count], 'b', label=f'Iteration {count}')
        count += 1
    
    axes[i].set_xlabel('Loss Index')
    axes[i].set_ylabel('Loss Value')
    axes[i].set_title(f'Permutation {i+1} - All Iterations')
    axes[i].legend()

plt.tight_layout()  # Adjust subplot parameters to fit into the figure
plt.show()
