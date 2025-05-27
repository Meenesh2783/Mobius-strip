import numpy as np
import matplotlib.pyplot as plt
# mpl_toolkits.mplot3d is implicitly used by projection='3d'
# from mpl_toolkits.mplot3d import Axes3D # No longer strictly necessary for recent matplotlib

class MobiusStrip:
    """
    Models a Mobius strip using parametric equations and computes geometric properties.
    """

    def __init__(self, R, w, resolution):
        """
        Initializes the MobiusStrip.

        Args:
            R (float): Radius of the center line of the strip.
            w (float): Width of the strip.
            resolution (int): Number of points along the u and v parameters
                             for mesh generation. (n x n grid of parameters)
        """
        if R <= 0:
            raise ValueError("R (radius) must be positive.")
        if w <= 0:
            raise ValueError("w (width) must be positive.")
        if w >= 2 * R:
            # This is a soft warning, as self-intersection is complex for Mobius.
            # A true check would be harder. This is a simple heuristic.
            print("Warning: Width 'w' is large relative to radius 'R'. "
                  "The strip might self-intersect visually.")
        if resolution < 3:
            raise ValueError("Resolution must be at least 3 for meaningful mesh.")

        self.R = R
        self.w = w
        self.resolution_u = resolution  # Points along u (circumference)
        self.resolution_v = resolution  # Points along v (width)

        # Placeholder for computed properties
        self.X = None
        self.Y = None
        self.Z = None
        self.surface_area = 0.0
        self.edge_length = 0.0

        self._generate_mesh()
        self._compute_surface_area()
        self._compute_edge_length()

    def _parametric_equations(self, u, v):
        """
        Computes x, y, z coordinates for given u, v parameters.

        Args:
            u (float or np.ndarray): Parameter u, typically in [0, 2*pi].
            v (float or np.ndarray): Parameter v, typically in [-w/2, w/2].

        Returns:
            tuple: (x, y, z) coordinates.
        """
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def _generate_mesh(self):
        """
        Generates the 3D mesh points for the Mobius strip surface.
        """
        u_vals = np.linspace(0, 2 * np.pi, self.resolution_u)
        v_vals = np.linspace(-self.w / 2, self.w / 2, self.resolution_v)
        
        U, V = np.meshgrid(u_vals, v_vals)
        
        self.X, self.Y, self.Z = self._parametric_equations(U, V)
        # Note: For plot_surface, X, Y, Z should be (resolution_v, resolution_u)
        # If meshgrid order is u_vals, v_vals then X,Y,Z are (len(v_vals), len(u_vals))
        # which is (self.resolution_v, self.resolution_u). This is fine.

    def _compute_surface_area(self):
        """
        Numerically approximates the surface area of the Mobius strip.
        It sums the areas of small quadrilaterals (approximated as two triangles)
        formed by the mesh grid.
        """
        area = 0.0
        # Iterate over each quadrilateral in the mesh
        # self.X, self.Y, self.Z have shape (resolution_v, resolution_u)
        for i in range(self.resolution_v - 1):      # Iterate over v strips
            for j in range(self.resolution_u - 1):  # Iterate over u segments
                # Points of the quadrilateral
                p00 = np.array([self.X[i, j],   self.Y[i, j],   self.Z[i, j]])
                p10 = np.array([self.X[i, j+1], self.Y[i, j+1], self.Z[i, j+1]]) # Next u
                p01 = np.array([self.X[i+1, j], self.Y[i+1, j], self.Z[i+1, j]]) # Next v
                p11 = np.array([self.X[i+1,j+1],self.Y[i+1,j+1],self.Z[i+1,j+1]])

                # Divide quadrilateral into two triangles: (p00, p10, p01) and (p11, p01, p10)
                # Area of triangle = 0.5 * ||AB x AC||
                
                # Triangle 1 (p00, p10, p01)
                vec1_tri1 = p10 - p00
                vec2_tri1 = p01 - p00
                area += 0.5 * np.linalg.norm(np.cross(vec1_tri1, vec2_tri1))
                
                # Triangle 2 (p11, p01, p10)
                vec1_tri2 = p01 - p11
                vec2_tri2 = p10 - p11
                area += 0.5 * np.linalg.norm(np.cross(vec1_tri2, vec2_tri2))
                
        self.surface_area = area

    def _compute_edge_length(self):
        """
        Numerically approximates the length of the single edge of the Mobius strip.
        The edge is parameterized by fixing v = w/2 (or -w/2) and letting u
        go from 0 to 4*pi due to the half-twist making it a single continuous edge.
        """
        # Use a higher resolution for edge calculation for better accuracy
        num_edge_points = 2 * self.resolution_u  # Or a fixed larger number
        
        # The edge is a single curve. We trace it by letting u go from 0 to 4*pi
        # while keeping v at one of its extremes (e.g., w/2).
        u_edge = np.linspace(0, 4 * np.pi, num_edge_points)
        v_fixed = self.w / 2
        
        x_edge, y_edge, z_edge = self._parametric_equations(u_edge, v_fixed)
        
        edge_points = np.vstack((x_edge, y_edge, z_edge)).T
        
        # Calculate segment lengths and sum them
        segment_vectors = np.diff(edge_points, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        self.edge_length = np.sum(segment_lengths)

    def get_properties(self):
        """
        Returns the computed geometric properties.

        Returns:
            dict: A dictionary containing 'surface_area' and 'edge_length'.
        """
        return {
            "surface_area": self.surface_area,
            "edge_length": self.edge_length
        }

    def plot(self, ax=None, save_path=None):
        """
        Plots the 3D Mobius strip.

        Args:
            ax (matplotlib.axes.Axes3D, optional): An existing 3D axes object.
                If None, a new figure and axes are created.
            save_path (str, optional): If provided, saves the plot to this path.
        """
        if self.X is None or self.Y is None or self.Z is None:
            print("Mesh not generated. Call _generate_mesh() first or initialize object.")
            return

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()

        # plot_surface expects X, Y, Z to be 2D arrays
        # The default colormap 'viridis' is good. 'coolwarm' or 'magma' also work well.
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', 
                        edgecolor='k', linewidth=0.2, rstride=1, cstride=1, alpha=0.8)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Mobius Strip (R={self.R}, w={self.w}, res={self.resolution_u}x{self.resolution_v})")
        
        # Set aspect ratio to be somewhat equal, or at least visually pleasing
        # This is tricky for 3D plots, but we can try to set limits based on data range
        max_range = np.array([self.X.max()-self.X.min(), 
                              self.Y.max()-self.Y.min(), 
                              self.Z.max()-self.Z.min()]).max() / 2.0
        
        mid_x = (self.X.max()+self.X.min()) * 0.5
        mid_y = (self.Y.max()+self.Y.min()) * 0.5
        mid_z = (self.Z.max()+self.Z.min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.show()

        if save_path:
            fig.savefig(save_path)
            print(f"Plot saved to {save_path}")


# --- Main execution ---
if __name__ == "__main__":
    # Parameters for the Mobius strip
    R_val = 10.0  # Radius from center to the strip's centerline
    w_val =10.0   # Width of the strip
    n_res = 200    # Resolution (number of points along u and v for the mesh)

    print(f"Creating Mobius strip with R={R_val}, w={w_val}, resolution={n_res}x{n_res}")

    try:
        mobius = MobiusStrip(R=R_val, w=w_val, resolution=n_res)
        properties = mobius.get_properties()

        print("\nGeometric Properties:")
        print(f"  Approximate Surface Area: {properties['surface_area']:.4f} units^2")
        print(f"  Approximate Edge Length:  {properties['edge_length']:.4f} units")

        # Plotting (optional, but good for visualization)
        print("\nGenerating plot...")
        mobius.plot(save_path="mobius_strip_plot.png")
        # The plot will be displayed and then saved if save_path is provided.

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")