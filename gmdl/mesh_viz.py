import numpy as np

def displacement_to_color(displacements, cmap="plasma"):
    norm = (displacements - displacements.min()) / (displacements.max() + 1e-12)
    import matplotlib as mpl
    cmap = mpl.colormaps.get(cmap)
    return (cmap(norm)[:, :3] * 255).astype(np.uint8)

def export_colored_pointcloud_ply(points, colors, path):
    N = len(points)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for pt, col in zip(points, colors):
        lines.append(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(col[0])} {int(col[1])} {int(col[2])}")
    with open(path, "w") as f:
        f.write("\n".join(lines))

def export_gray_mesh_ply(vertices, faces, path, gray=180):
    faces = np.asarray(faces, dtype=int)
    N = len(vertices)
    M = len(faces)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {M}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for v in vertices:
        lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {gray} {gray} {gray}")
    for f in faces:
        lines.append(f"3 {f[0]} {f[1]} {f[2]}")
    with open(path, "w") as f:
        f.write("\n".join(lines))

def render_pointcloud_mpl(points, colors, displacement, path,
                          mesh_vertices=None, mesh_faces=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if mesh_vertices is not None and mesh_faces is not None:
        mesh_faces = np.asarray(mesh_faces, dtype=int)
        tri = []
        for f in mesh_faces:
            if f.min() >= 0 and f.max() < len(mesh_vertices):
                tri.append([mesh_vertices[i] for i in f])
        if tri:
            ax.add_collection3d(Poly3DCollection(
                tri, alpha=0.06, facecolor="gray", edgecolor="gray", linewidth=0.2,
            ))

    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=displacement, cmap="plasma", s=8, alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Vertex displacement after latent edit")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Displacement (mm)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
