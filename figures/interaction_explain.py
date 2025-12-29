import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    # X-axis values
    x = np.linspace(0, 1, 100)  # imaging measure from 0 to 1

    # Blue line (now the common one across all plots) - a simple decreasing linear function
    blue_line = 0.9 - 0.8 * x

    # Red lines for different plots
    red_line_parallel = 0.7 - 0.8 * x  # parallel to blue line, lower intercept
    red_line_cross = 0.4 - 0.2 * x  # less steep; crosses blue at ~x=0.6

    # Create 3 plots side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Plot 1: Single blue line
    axs[0].plot(x, blue_line, 'b')
    axs[0].set_title("Cohort study predicting outcome")
    axs[0].set_xlabel("Your (imaging) measure")
    axs[0].set_ylabel("Probability of good outcome")
    axs[0].set_ylim(0, 1)

    # Plot 2: Blue and red parallel decreasing lines
    axs[1].plot(x, blue_line, 'b', label='Intervention')
    axs[1].plot(x, red_line_parallel, 'r', label='Control')
    axs[1].set_title("No treatment effect modification")
    axs[1].set_xlabel("Your (imaging) measure")
    axs[1].set_ylim(0, 1)

    # Plot 3: Blue and red, crossing at ~0.6
    axs[2].plot(x, blue_line, 'b', label='Intervention')
    axs[2].plot(x, red_line_cross, 'r', label='Control')
    axs[2].set_title("Treatment effect modification")
    axs[2].set_xlabel("Your (imaging) measure")
    axs[2].set_ylim(0, 1)

    # Add legends
    for ax in axs:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('/home/hvv/Documents/projects/BL_NCCT/interaction_explain.png')
    plt.show()


    print(1)