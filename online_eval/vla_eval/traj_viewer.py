import sys
import numpy as np
import magnum as mn
import habitat_sim
# Import the base application you provided (assuming it's saved as interactive_viewer.py)
from interactive_viewer import HabitatSimInteractiveViewer, default_sim_settings

class TrajectoryViewer(HabitatSimInteractiveViewer):
    def __init__(self, sim_settings, ref_traj, pred_traj, radius=0.06):
        """
        Args:
            sim_settings: Standard Habitat-Sim configuration dict.
            ref_traj: List or array of [x, y, z] points for the Ground Truth.
            pred_traj: List or array of [x, y, z] points for the Model Prediction.
            radius: The thickness of the rendered trajectory tubes.
        """
        self.ref_traj = ref_traj
        self.pred_traj = pred_traj
        self.traj_radius = radius
        
        # Initialize the heavy-lifting Magnum application
        super().__init__(sim_settings)

    def reconfigure_sim(self) -> None:
        """
        Overrides the setup method to inject our trajectories 
        right after the simulator initializes the 3D scene.
        """
        # 1. Let the original viewer load the scene and setup the agent
        super().reconfigure_sim()

        # 2. Inject our custom trajectory visualisations
        self._draw_trajectories()

    def _draw_trajectories(self):
        """Builds the 3D tubes using Habitat's API."""
        if not self.sim:
            return

        # ── DRAW REFERENCE TRAJECTORY (Solid Cyan) ─────────────
        if self.ref_traj is not None and len(self.ref_traj) > 1:
            # Convert python/numpy floats to Magnum Vector3s
            ref_points = [
                mn.Vector3(float(p[0]), float(p[1]), float(p[2])) 
                for p in self.ref_traj
            ]

            # Use add_trajectory_object for a solid reference line
            traj_id = self.sim.add_trajectory_object(
                traj_vis_name="reference_trajectory",
                points=ref_points,
                num_segments=8,
                radius=self.traj_radius,
                color=mn.Color4(0.0, 1.0, 1.0, 1.0), # Cyan
                smooth=True,
                num_interpolations=10
            )
            print(f"✓ Added Reference Trajectory (ID: {traj_id})")

        # ── DRAW PREDICTED TRAJECTORY (Gradient: Green to Red) ──
        if self.pred_traj is not None and len(self.pred_traj) > 1:
            pred_points = [
                mn.Vector3(float(p[0]), float(p[1]), float(p[2])) 
                for p in self.pred_traj
            ]

            # Generate a color gradient to show progression
            pred_colors = []
            n = len(pred_points)
            for i in range(n):
                t = i / max(n - 1, 1) # 0.0 at start, 1.0 at end
                # Color3 format: (Red, Green, Blue). Green fades out, Red fades in.
                pred_colors.append(mn.Color3(t, 1.0 - t, 0.0))

            # Use add_gradient_trajectory_object for the predicted path
            traj_id = self.sim.add_gradient_trajectory_object(
                traj_vis_name="predicted_trajectory",
                points=pred_points,
                colors=pred_colors,
                num_segments=8,
                radius=self.traj_radius,
                smooth=True,
                num_interpolations=10
            )
            print(f"✓ Added Predicted Trajectory (ID: {traj_id})")


# ==========================================================
# Run Execution
# ==========================================================
if __name__ == "__main__":
    import argparse
    
    # Example Mock Data: Replace with your actual coordinate loading logic
    # Make sure coordinates are already transformed to Habitat's Y-Up system!
    mock_ref_traj = [
        [0.0, 1.0, 0.0], [0.0, 1.0, -1.0], [1.0, 1.0, -1.0]
    ]
    mock_pred_traj = [
        [0.0, 1.0, 0.0], [0.1, 1.0, -0.8], [0.8, 1.0, -0.9]
    ]

    # Setup standard simulation settings
    sim_settings = default_sim_settings.copy()
    sim_settings["scene"] = "/home/testunot/datasets/habitat/IndoorUAV-VLA/scene_datasets/gibson/Adrian.glb" # Replace with your test GLB
    sim_settings["window_width"] = 1280
    sim_settings["window_height"] = 720
    sim_settings["enable_physics"] = True # Set to True if testing physical collisions
    
    # Start the viewer
    viewer = TrajectoryViewer(
        sim_settings=sim_settings,
        ref_traj=mock_ref_traj,
        pred_traj=mock_pred_traj,
        radius=0.05
    )
    
    print("Starting Interactive Viewer... Press ESC to exit.")
    viewer.exec()