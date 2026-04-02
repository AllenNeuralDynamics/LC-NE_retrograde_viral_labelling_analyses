import matplotlib.pyplot as plt
from ipywidgets import interactive, widgets


def plot_sections(roi_mask, ML, RC, DV, ML_min, ML_max, RC_min, RC_max, DV_min, DV_max):
    """Used by interactive_plot_sections to show roi_mask"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Coronal section (axis=0)
    coronal_slice = roi_mask[RC, :, :]  # Slice along RC
    axes[0].imshow(coronal_slice, interpolation="none", alpha=0.3)
    axes[0].vlines(ML, DV_min, DV_max, colors="gray", linestyles="dashed")
    axes[0].hlines(DV, ML_min, ML_max, colors="gray", linestyles="dashed")
    axes[0].set(xlabel="ML", ylabel="DV", title="Coronal section", xlim=(ML_min, ML_max), ylim=(DV_max, DV_min))

    # Transverse section (axis=1)
    transverse_slice = roi_mask[:, DV, :]  # Slice along DV
    axes[1].imshow(transverse_slice, interpolation="none", alpha=0.3)
    axes[1].vlines(ML, RC_min, RC_max, colors="gray", linestyles="dashed")
    axes[1].hlines(RC, ML_min, ML_max, colors="gray", linestyles="dashed")
    axes[1].set(title="Transverse section", xlabel="ML", ylabel="RC", xlim=(ML_min, ML_max), ylim=(RC_max, RC_min))

    # Sagittal section (axis=2)
    sagittal_slice = roi_mask[:, :, ML]  # Slice along ML
    axes[2].imshow(sagittal_slice, interpolation="none", alpha=0.3)
    axes[2].vlines(DV, RC_min, RC_max, colors="gray", linestyles="dashed")
    axes[2].hlines(RC, DV_min, DV_max, colors="gray", linestyles="dashed")
    axes[2].set(title="Sagittal section", xlabel="DV", ylabel="RC", xlim=(DV_min, DV_max), ylim=(RC_max, RC_min))

    plt.tight_layout()
    plt.show()


# Function to wrap interactive widget
def interactive_plot_mask(roi_mask, MLinit, RCinit, DVinit, ML_min, ML_max, RC_min, RC_max, DV_min, DV_max):
    interactive_plot = interactive(
        lambda ML, RC, DV: plot_sections(roi_mask, ML, RC, DV, ML_min, ML_max, RC_min, RC_max, DV_min, DV_max),
        ML=widgets.IntSlider(min=ML_min, max=ML_max - 1, step=1, value=MLinit, description="ML:"),
        RC=widgets.IntSlider(min=RC_min, max=RC_max - 1, step=1, value=RCinit, description="RC:"),
        DV=widgets.IntSlider(min=DV_min, max=DV_max - 1, step=1, value=DVinit, description="DV:"),
    )
    display(interactive_plot)
