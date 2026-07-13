import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt


def format_vectors_from_flat(u: np.ndarray, n_dim: int = 2) -> np.ndarray:
    u_x = []
    u_y = []
    u_z = []
    for u_i in u:
        u_x.append(u_i[::n_dim])
        u_y.append(u_i[1::n_dim])
        if n_dim == 3:
            u_z.append(u_i[2::n_dim])

    if n_dim == 2:
        u_z = [[0] * len(u_x[0])] * len(u_x)

    return np.array([u_x, u_y, u_z]).transpose(1, 2, 0)


def create_mesh_animation(mesh, scalars=None, vectors=None, name: str = "result", colormap: str = "viridis", color_limits: list = None, **kwargs) -> None:
    if scalars is None and vectors is None:
        raise ValueError("Either scalars or vectors must be provided")

    show_scalar_bar = True

    time_steps = 0

    if scalars is None:
        show_scalar_bar = False
        scalars = np.zeros((len(vectors), len(vectors[0])))
        time_steps = len(vectors)
    elif vectors is None:
        vectors = np.zeros((len(scalars), len(scalars[0]), 3))
        time_steps = len(scalars)
    else:
        time_steps = len(scalars)

    if color_limits is None:
        color_limits = [scalars.min(), scalars.max()]
    else:
        if color_limits[0] is None:
            color_limits[0] = scalars.min()
        if color_limits[1] is None:
            color_limits[1] = scalars.max()

    cmap = plt.get_cmap(colormap, 25)
    sargs = dict(
        title_font_size=25,
        label_font_size=20,
        fmt="%.2e",
        color="black",
        position_x=0.1,
        position_y=0.8,
        width=0.8,
        height=0.1,
    )

    grid_props = {
        "show_edges": True,
        "lighting": False,
        "cmap": cmap,
        "scalar_bar_args": sargs,
        "show_scalar_bar": show_scalar_bar,
        "clim": color_limits,
        "name": "grid",
    }

    grid_base = pv.UnstructuredGrid(*mesh)
    grid_base['scalars'] = scalars[0]
    grid_base.set_active_scalars('scalars')
    grid_base['vectors'] = vectors[0]
    grid_base.set_active_vectors('vectors')
    grid = grid_base.warp_by_vector()

    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(f"{name}.gif")

    plotter.add_mesh(
        grid,
        **grid_props,
    )

    text = plotter.add_text(text=f"0/{time_steps}", position="lower_left", font_size=8)

    if np.any(mesh[2][:, 2] != 0):
        plotter.view_isometric()
    else:
        plotter.view_xy()
    plotter.camera.zoom(1.3)

    for i, (scalar, vector) in enumerate(zip(scalars, vectors)):
        grid_base['vectors'] = vector
        grid = grid_base.warp_by_vector()
        grid['scalars'] = scalar
        plotter.add_mesh(
            grid,
            **grid_props
        )
        text.SetText(0, f"{i}/{time_steps - 1}")
        plotter.write_frame()

    plotter.close()
