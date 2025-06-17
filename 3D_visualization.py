from vedo import load
from vedo.applications import RayCastPlotter, IsosurfaceBrowser

def visualize_volumes(data_in):
    # Load the NIfTI file
    vol = load(data_in + '/volume.nii')

    # Show the segmented volume
    # Ray Casting
    plt = RayCastPlotter(vol, bg='black', bg2='blackboard', axes=7)  # Plotter instance
    plt.show(viewup="z").close()

    # Marching Cubes
    plt = IsosurfaceBrowser(vol, use_gpu=True, c='gold') # Plotter instance
    plt.show(axes=7, bg2='lb').close()

    # Load the NIfTI file
    vol2 = load(data_in + '/segmentation.nii')

    # Show the segmented volume
    # Ray Casting
    plt = RayCastPlotter(vol2, bg='black', bg2='blackboard', axes=7)  # Plotter instance
    plt.show(viewup="z").close()

    # Marching Cubes
    plt = IsosurfaceBrowser(vol2, use_gpu=True, c='gold') # Plotter instance
    plt.show(axes=7, bg2='lb').close()

if __name__ == '__main__':
    visualize_volumes('volumes')