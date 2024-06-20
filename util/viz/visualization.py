import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def plot_two_variables(df, var1, var2, binary_column=None) -> Figure:
    """
    Plots a scatter plot for two variables of a DataFrame, coloring the points according to a binary column if provided.

    Parameters:
        - df: Pandas DataFrame containing the data.
        - var1: Name of the first variable (x-axis).
        - var2: Name of the second variable (y-axis).
        - binary_column: Name of the binary column used to color the points (optional).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if binary_column is not None:
        # Getting the unique values of the binary column
        classes = df[binary_column].unique()

        # Creating a class-to-color mapping
        colors = {cls: plt.cm.tab10(i) for i, cls in enumerate(classes)}

        # Coloring the points according to the classes
        for cls in classes:
            df_cls = df[df[binary_column] == cls]
            ax.scatter(df_cls[var1], df_cls[var2], color=colors[cls], label=cls, alpha=0.5)
        
        ax.legend(title=binary_column)
    else:
        ax.scatter(df[var1], df[var2], alpha=0.5)

    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_title(f'Scatter Plot: {var1} vs {var2}')
    ax.grid(True)
    plt.tight_layout()
    return fig
