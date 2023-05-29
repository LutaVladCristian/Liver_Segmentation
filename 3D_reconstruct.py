"""
Created on Thu Apr  6 14:29:08 2023

@author: vlad_cristian.luta
"""

from vedo import load
from vedo.applications import IsosurfaceBrowser, RayCastPlotter

# Load the NIfTI file
vol = load('examples/liver_101_0.nii.gz')

# Show the volume
vol.show()

#Ray Casting
plt = RayCastPlotter(vol, bg='black', bg2='blackboard', axes=7)  # Plotter instance
plt.show(viewup="z").close()

#Marching Cubes
plt = IsosurfaceBrowser(vol, c='gold')
plt.show(axes=7, bg2='lb').close()