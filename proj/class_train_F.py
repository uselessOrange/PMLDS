import numpy as np
import matplotlib.pyplot as plt


class ModelEvaluator:
    def __init__(self, permutations, num_iterations):
        self.permutations = permutations
        self.num_iterations = num_iterations
        self.losses_per_permutation = []
        self.val_losses_per_permutation = []
        self.rmse_matrix = np.zeros([3, num_iterations, permutations])
        
        
    def evaluate_models(self):
        for permutation in range(self.permutations):
            losses = []
            val_losses = []
            for j in range(self.num_iterations):
                size = np.random.randint(5, 15)
                loss = np.random.normal(1, 2, size=size)
                losses.append(loss)
                val_losses.append(-loss)
                
                self.rmse_matrix[0, j, permutation] = np.random.normal(1, 1000)
                self.rmse_matrix[1, j, permutation] = np.random.normal(1, 1000)
                self.rmse_matrix[2, j, permutation] = np.random.normal(1, 1000)
            self.losses_per_permutation.append(losses)
            self.val_losses_per_permutation.append(val_losses)
    
    def plot_data(self):
        fig, axes = plt.subplots(len(self.losses_per_permutation), 1, figsize=(8, 6 * len(self.losses_per_permutation)))

        # Plotting all iterations for each permutation as subplots
        for i, permutation_losses in enumerate(self.losses_per_permutation):
            count = 0
            for loss in permutation_losses:
                axes[i].plot(np.arange(len(loss)), loss, 'r', label=f'Iteration {count}')
                axes[i].plot(np.arange(len(self.val_losses_per_permutation[i][count])), self.val_losses_per_permutation[i][count], 'b', label=f'Iteration {count}')
                count += 1

            axes[i].set_xlabel('Loss Index')
            axes[i].set_ylabel('Loss Value')
            axes[i].set_title(f'Permutation {i+1} - All Iterations')
            axes[i].legend()

        plt.tight_layout()  # Adjust subplot parameters to fit into the figure
        plt.show()

    def plot_box(self):
        #Create subplots for each set (training, testing, validation)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        # Plot boxplots for each set (training, testing, validation)
        for i, set_name in enumerate(['Training', 'Testing', 'Validation']):
            # Data for the current set
            data = self.rmse_matrix[i]
            
            # Plot boxplot for each permutation
            axs[i].boxplot(data.T)
            axs[i].set_title(f'{set_name} Set')
            axs[i].set_xlabel('Permutation')
            axs[i].set_ylabel('RMSE Value')

        plt.tight_layout()
        plt.show()
# Usage example:
evaluator = ModelEvaluator(permutations=3, num_iterations=10)
evaluator.evaluate_models()
evaluator.plot_data()
evaluator.plot_box()
