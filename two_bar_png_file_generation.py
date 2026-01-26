import os
import matplotlib.pyplot as plt
import math

# -----------------------------
# Configuration
# -----------------------------
output_dir = "2_bar_chart_output"
output_file = "two_bar_chart.png"

labels = ["A", "B"]
values = [37, 62]
n_ticks = 5

# -----------------------------
# Ensure output directory exists
# -----------------------------
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Create bar chart
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(labels, values)

# Integer y-axis limits
y_min = 0
y_max = math.ceil(max(values) * 1.1)
ax.set_ylim(y_min, y_max)

# Integer tick values
tick_values = list(range(0, y_max + 1, y_max // (n_ticks - 1)))
ax.set_yticks(tick_values)

# Labels and title
ax.set_ylabel("Data Value")
ax.set_title("Two-Bar Chart")

# -----------------------------
# Save figure
# -----------------------------
save_path = os.path.join(output_dir, output_file)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Saved bar chart to: {save_path}")
