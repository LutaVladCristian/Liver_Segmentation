from vedo import load, Plotter
from vedo.applications import RayCastPlotter, IsosurfaceBrowser

def visualize_volumes(data_in):
    # Load the NIfTI volume
    vol = load(data_in + '/segmentation.nii')

    # Visualize with RayCasting (raw volume)
    plt = RayCastPlotter(vol, bg='black', bg2='blackboard', axes=7)
    plt.show(viewup="z").close()

    # Extract surface with marching cubes
    surface = vol.isosurface(0.5)  # adjust threshold if needed

    # Smooth the mesh
    surface_smooth = surface.clone().smoothWSinc(niter=50, pass_band=0.1)

    # Show smoothed surface (as a mesh, not a browser)
    vp = Plotter(title="Smoothed Mesh", axes=7, bg2='lb')
    vp.show(surface_smooth, viewup="z").close()

visualize_volumes('volumes')