# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:29:11 2026

@author: admin
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ------------------------------------------------------------
# Profile catalogue
# area_mm2: cross-sectional area
# density: kg/m^3
# E: Young's modulus in Pa
# cost_per_kg: estimated material cost in €/kg
# ------------------------------------------------------------

PROFILE_CATALOGUE = {
    "Flat 20x5": {
        "area_mm2": 100.0,
        "E": 210e9,
        "density": 7850.0,
        "cost_per_kg": 2.50,
    },
    "Flat 30x5": {
        "area_mm2": 150.0,
        "E": 210e9,
        "density": 7850.0,
        "cost_per_kg": 2.50,
    },
    "Round d10": {
        "area_mm2": np.pi * 10.0**2 / 4.0,
        "E": 210e9,
        "density": 7850.0,
        "cost_per_kg": 2.80,
    },
    "Round d15": {
        "area_mm2": np.pi * 15.0**2 / 4.0,
        "E": 210e9,
        "density": 7850.0,
        "cost_per_kg": 2.80,
    },
}


# ------------------------------------------------------------
# Core truss solver
# ------------------------------------------------------------

def solve_truss(nodes_df, members_df, supports_df, loads_df):
    """
    Linear-elastic 2D truss solver.

    Assumptions:
    - pin-jointed nodes
    - axial bar elements only
    - small displacements
    - linear elastic material
    - support displacements are zero
    """

    nodes_df = nodes_df.copy()
    members_df = members_df.copy()
    supports_df = supports_df.copy()
    loads_df = loads_df.copy()

    node_ids = list(nodes_df["node"])
    node_to_index = {node: i for i, node in enumerate(node_ids)}

    n_nodes = len(nodes_df)
    n_dof = 2 * n_nodes

    coords = nodes_df[["x", "y"]].to_numpy(dtype=float)

    K = np.zeros((n_dof, n_dof), dtype=float)
    F = np.zeros(n_dof, dtype=float)

    member_results = []

    # Assemble global stiffness matrix
    for _, member in members_df.iterrows():
        member_id = member["member"]
        ni = member["node_i"]
        nj = member["node_j"]
        profile_name = member["profile"]

        if ni not in node_to_index or nj not in node_to_index:
            raise ValueError(f"Member {member_id}: unknown node.")
        if profile_name not in PROFILE_CATALOGUE:
            raise ValueError(f"Member {member_id}: unknown profile '{profile_name}'.")

        i = node_to_index[ni]
        j = node_to_index[nj]

        xi, yi = coords[i]
        xj, yj = coords[j]

        dx = xj - xi
        dy = yj - yi
        L = np.sqrt(dx**2 + dy**2)

        if L <= 0:
            raise ValueError(f"Member {member_id}: length is zero.")

        c = dx / L
        s = dy / L

        profile = PROFILE_CATALOGUE[profile_name]
        A = profile["area_mm2"] * 1e-6
        E = profile["E"]

        k_local_global = (E * A / L) * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s],
        ])

        dofs = np.array([2*i, 2*i + 1, 2*j, 2*j + 1])

        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += k_local_global[a, b]

        member_results.append({
            "member": member_id,
            "node_i": ni,
            "node_j": nj,
            "profile": profile_name,
            "length_m": L,
            "c": c,
            "s": s,
            "A_m2": A,
            "E_Pa": E,
            "density_kg_m3": profile["density"],
            "cost_per_kg": profile["cost_per_kg"],
        })

    # Assemble external loads
    for _, load in loads_df.iterrows():
        node = load["node"]
        if node not in node_to_index:
            raise ValueError(f"Load: unknown node '{node}'.")

        i = node_to_index[node]
        F[2*i] += float(load["Fx"])
        F[2*i + 1] += float(load["Fy"])

    # Boundary conditions
    fixed_dofs = []

    for _, support in supports_df.iterrows():
        node = support["node"]
        if node not in node_to_index:
            raise ValueError(f"Support: unknown node '{node}'.")

        i = node_to_index[node]

        if bool(support["fix_x"]):
            fixed_dofs.append(2*i)
        if bool(support["fix_y"]):
            fixed_dofs.append(2*i + 1)

    fixed_dofs = sorted(set(fixed_dofs))
    free_dofs = [dof for dof in range(n_dof) if dof not in fixed_dofs]

    if len(free_dofs) == 0:
        raise ValueError("No free degrees of freedom available.")

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]

    # Check solvability
    rank = np.linalg.matrix_rank(K_ff)
    if rank < K_ff.shape[0]:
        raise ValueError(
            "The reduced stiffness matrix is singular. "
            "The structure is probably unstable, insufficiently supported, "
            "or contains a mechanism."
        )

    # Solve displacements
    u = np.zeros(n_dof, dtype=float)
    u[free_dofs] = np.linalg.solve(K_ff, F_f)

    # Reactions
    R = K @ u - F

    # Member forces, stresses, masses, costs
    member_output = []

    for result in member_results:
        ni = result["node_i"]
        nj = result["node_j"]

        i = node_to_index[ni]
        j = node_to_index[nj]

        dofs = np.array([2*i, 2*i + 1, 2*j, 2*j + 1])
        u_e = u[dofs]

        c = result["c"]
        s = result["s"]
        L = result["length_m"]
        E = result["E_Pa"]
        A = result["A_m2"]

        extension = np.array([-c, -s, c, s]) @ u_e
        axial_force = E * A / L * extension
        stress = axial_force / A

        mass = A * L * result["density_kg_m3"]
        cost = mass * result["cost_per_kg"]

        member_output.append({
            "member": result["member"],
            "node_i": ni,
            "node_j": nj,
            "profile": result["profile"],
            "length_m": L,
            "axial_force_N": axial_force,
            "type": "tension" if axial_force >= 0 else "compression",
            "stress_MPa": stress / 1e6,
            "mass_kg": mass,
            "cost_EUR": cost,
        })

    # Displacement table
    displacement_output = []

    for node in node_ids:
        i = node_to_index[node]
        displacement_output.append({
            "node": node,
            "ux_mm": u[2*i] * 1000.0,
            "uy_mm": u[2*i + 1] * 1000.0,
        })

    # Reaction table
    reaction_output = []

    for node in node_ids:
        i = node_to_index[node]
        reaction_output.append({
            "node": node,
            "Rx_N": R[2*i],
            "Ry_N": R[2*i + 1],
        })

    return {
        "nodes": nodes_df,
        "members": pd.DataFrame(member_output),
        "displacements": pd.DataFrame(displacement_output),
        "reactions": pd.DataFrame(reaction_output),
        "K": K,
        "F": F,
        "u": u,
    }


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_truss(nodes_df, members_df, results=None, scale=1.0):
    fig = go.Figure()

    node_to_xy = {
        row["node"]: (float(row["x"]), float(row["y"]))
        for _, row in nodes_df.iterrows()
    }

    # undeformed structure
    for _, member in members_df.iterrows():
        ni = member["node_i"]
        nj = member["node_j"]

        if ni not in node_to_xy or nj not in node_to_xy:
            continue

        xi, yi = node_to_xy[ni]
        xj, yj = node_to_xy[nj]

        fig.add_trace(go.Scatter(
            x=[xi, xj],
            y=[yi, yj],
            mode="lines",
            line=dict(width=3),
            name=f"Member {member['member']}",
            showlegend=False,
            hovertext=f"Member {member['member']}: {ni}–{nj}",
        ))

    # nodes
    fig.add_trace(go.Scatter(
        x=[node_to_xy[n][0] for n in node_to_xy],
        y=[node_to_xy[n][1] for n in node_to_xy],
        mode="markers+text",
        text=list(node_to_xy.keys()),
        textposition="top center",
        marker=dict(size=10),
        name="Nodes",
    ))

    # deformed structure
    if results is not None:
        disp = results["displacements"]
        disp_map = {
            row["node"]: (row["ux_mm"] / 1000.0, row["uy_mm"] / 1000.0)
            for _, row in disp.iterrows()
        }

        for _, member in members_df.iterrows():
            ni = member["node_i"]
            nj = member["node_j"]

            if ni not in node_to_xy or nj not in node_to_xy:
                continue

            xi, yi = node_to_xy[ni]
            xj, yj = node_to_xy[nj]

            uxi, uyi = disp_map[ni]
            uxj, uyj = disp_map[nj]

            fig.add_trace(go.Scatter(
                x=[xi + scale * uxi, xj + scale * uxj],
                y=[yi + scale * uyi, yj + scale * uyj],
                mode="lines",
                line=dict(width=2, dash="dash"),
                name="Deformed",
                showlegend=False,
            ))

    fig.update_layout(
        title="2D Truss",
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        yaxis_scaleanchor="x",
        height=600,
    )

    return fig

def show_header_image(filename, caption=None):
    image_path = Path(__file__).parent / "assets" / filename

    if image_path.exists():
        st.image(
            image_path,
            caption=caption,
            width="stretch",
        )
    else:
        st.info(f"Header image not found: {image_path}")

# ------------------------------------------------------------
# Streamlit app
# ------------------------------------------------------------

st.set_page_config(page_title="Trust the Truss", layout="wide")

show_header_image("trust_the_truss_header.png")

st.title("Trust the Truss")
# st.title("2D Truss Solver")
st.caption("An easy-to-use tool for analyzing planar trusses with profile selection, mass calculation, and cost estimation.")
# st.caption("Linear-elastic planar truss analysis with profile selection, reactions, member forces, mass and cost estimate.")

with st.expander("How to use the user interface", expanded=False):
    st.markdown(
        """
        This app calculates support reactions, nodal displacements, member forces, stresses, material mass, and estimated material costs for planar truss structures.

        The tables already contain a small example truss. These predefined entries are only intended as a guide and can be overwritten. You can change the names of nodes and members directly in the tables, for example by replacing `A`, `B`, `C` or `S1`, `S2`, `S3` with your own labels.

        **Nodes**  
        Enter the node coordinates in the node table. Each node requires a unique name and its Cartesian coordinates `x` and `y`.

        **Members**  
        Define each truss member by specifying its start node `node_i`, its end node `node_j`, and a profile from the profile list. The node names must match the names used in the node table.

        **Supports**  
        Define the support conditions by selecting which translational degrees of freedom are fixed. Use `fix_x` for a restrained horizontal displacement and `fix_y` for a restrained vertical displacement.

        **External loads**  
        Enter external nodal forces in the load table. The force components `Fx` and `Fy` are applied at the specified node.

        **Adding new entries**  
        To add new nodes, members, supports, or loads, click the `+` symbol in the bottom row 
        of the corresponding table. You can also overwrite existing rows if you want to replace the example structure completely.

        **Profile selection**  
        Each member can be assigned a profile from the predefined profile catalogue. The selected profile is used for stiffness, stress, mass, and cost calculations.

        **Calculation**  
        After entering the complete model, click **Calculate truss**. The app will compute and display the results below the input section.

        **Deformation plot**  
        The deformation scale slider changes only the graphical magnification of the displayed deformed shape. It does not change the actual calculation results.
        
        **Model assumptions**  
        The current version assumes pin-jointed nodes, axial member forces only, small  displacements, linear-elastic material behavior, and external loads acting only at nodes.  Bending moments, shear forces, distributed loads, and buckling checks are not included.
        
        """
    )

with st.expander("License, attribution, and disclaimer", expanded=False):
    st.markdown(
        """
        **Copyright**  
        © 2026 Thorsten Bartel.

        **License**  
        The source code of this application is released under the MIT License, unless otherwise stated. Please refer to the accompanying `LICENSE` file for details.

        **AI-assisted development**  
        The concept, user interface structure, and parts of the Python/Streamlit implementation were developed with assistance from OpenAI ChatGPT, GPT-5.5 Thinking, June 2026.

        **Third-party software**  
        This app is implemented in Python using Streamlit, NumPy, pandas, and Plotly. These packages are distributed under their respective open-source licenses.

        **Disclaimer**  
        This app is intended for teaching and demonstration purposes only. It is based on a simplified linear-elastic planar truss model and does not replace professional engineering verification, safety assessment, or code-compliant structural design.
        """
    )

# with st.expander("Model assumptions", expanded=False):
#     st.write(
#         """
#         This prototype assumes:
#         - pin-jointed truss nodes,
#         - axial bar forces only,
#         - no bending moments,
#         - no shear forces,
#         - no distributed loads,
#         - small displacements,
#         - linear-elastic material behavior.
#         """
#     )

# Default example: simple triangular truss
default_nodes = pd.DataFrame({
    "node": ["A", "B", "C"],
    "x": [0.0, 4.0, 2.0],
    "y": [0.0, 0.0, 3.0],
})

default_members = pd.DataFrame({
    "member": ["S1", "S2", "S3"],
    "node_i": ["A", "B", "A"],
    "node_j": ["C", "C", "B"],
    "profile": ["Flat 20x5", "Flat 20x5", "Flat 30x5"],
})

default_supports = pd.DataFrame({
    "node": ["A", "B"],
    "fix_x": [True, False],
    "fix_y": [True, True],
})

default_loads = pd.DataFrame({
    "node": ["C"],
    "Fx": [0.0],
    "Fy": [-1000.0],
})

left, right = st.columns([1, 1])

with left:
    st.subheader("Input data")

    st.write("### Nodes")
    nodes_df = st.data_editor(
        default_nodes,
        num_rows="dynamic",
        width="stretch",
    )

    st.write("### Members")
    members_df = st.data_editor(
        default_members,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "profile": st.column_config.SelectboxColumn(
                "profile",
                options=list(PROFILE_CATALOGUE.keys()),
            )
        },
    )

    st.write("### Supports")
    supports_df = st.data_editor(
        default_supports,
        num_rows="dynamic",
        width="stretch",
    )

    st.write("### External loads")
    loads_df = st.data_editor(
        default_loads,
        num_rows="dynamic",
        width="stretch",
    )

with right:
    st.subheader("Profile catalogue")

    profile_table = pd.DataFrame(PROFILE_CATALOGUE).T.reset_index()
    profile_table = profile_table.rename(columns={"index": "profile"})
    st.dataframe(profile_table, width="stretch")

    st.subheader("Visualization")
    st.plotly_chart(
        plot_truss(nodes_df, members_df),
        width="stretch",
    )

# if st.button("Calculate truss", type="primary"):
    
# ------------------------------------------------------------
# Calculation and persistent results
# ------------------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results = None

if "calculation_error" not in st.session_state:
    st.session_state.calculation_error = None


if st.button("Calculate truss", type="primary"):
    try:
        st.session_state.results = solve_truss(
            nodes_df,
            members_df,
            supports_df,
            loads_df,
        )
        st.session_state.calculation_error = None

    except Exception as e:
        st.session_state.results = None
        st.session_state.calculation_error = str(e)


if st.session_state.calculation_error is not None:
    st.error(st.session_state.calculation_error)


if st.session_state.results is not None:
    results = st.session_state.results

    st.success("Calculation completed.")

    st.subheader("Deformed structure")

    deformation_scale = st.slider(
        "Deformation scale factor",
        min_value=1.0,
        max_value=1000.0,
        value=100.0,
        step=10.0,
        key="deformation_scale",
    )

    st.plotly_chart(
        plot_truss(
            nodes_df,
            members_df,
            results=results,
            scale=deformation_scale,
        ),
        width="stretch",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Support reactions")
        reactions = results["reactions"].copy()
        reactions["Rx_N"] = reactions["Rx_N"].round(3)
        reactions["Ry_N"] = reactions["Ry_N"].round(3)
        st.dataframe(reactions, width="stretch")

    with col2:
        st.subheader("Nodal displacements")
        displacements = results["displacements"].copy()
        displacements["ux_mm"] = displacements["ux_mm"].round(6)
        displacements["uy_mm"] = displacements["uy_mm"].round(6)
        st.dataframe(displacements, width="stretch")

    st.subheader("Member forces, mass and cost")

    members_out = results["members"].copy()
    members_out["length_m"] = members_out["length_m"].round(3)
    members_out["axial_force_N"] = members_out["axial_force_N"].round(3)
    members_out["stress_MPa"] = members_out["stress_MPa"].round(3)
    members_out["mass_kg"] = members_out["mass_kg"].round(3)
    members_out["cost_EUR"] = members_out["cost_EUR"].round(2)

    st.dataframe(members_out, width="stretch")

    total_mass = results["members"]["mass_kg"].sum()
    total_cost = results["members"]["cost_EUR"].sum()

    st.metric("Total material mass", f"{total_mass:.3f} kg")
    st.metric("Estimated material cost", f"{total_cost:.2f} €")
        
    # ###########
    # try:
    #     results = solve_truss(nodes_df, members_df, supports_df, loads_df)

    #     st.success("Calculation completed.")

    #     st.subheader("Deformed structure")
    #     deformation_scale = st.slider(
    #         "Deformation scale factor",
    #         min_value=1.0,
    #         max_value=1000.0,
    #         value=100.0,
    #         step=10.0,
    #     )

    #     st.plotly_chart(
    #         plot_truss(nodes_df, members_df, results=results, scale=deformation_scale),
    #         width="stretch",
    #     )

    #     col1, col2 = st.columns(2)

    #     with col1:
    #         st.subheader("Support reactions")
    #         reactions = results["reactions"].copy()
    #         reactions["Rx_N"] = reactions["Rx_N"].round(3)
    #         reactions["Ry_N"] = reactions["Ry_N"].round(3)
    #         st.dataframe(reactions, width="stretch")

    #     with col2:
    #         st.subheader("Nodal displacements")
    #         displacements = results["displacements"].copy()
    #         displacements["ux_mm"] = displacements["ux_mm"].round(6)
    #         displacements["uy_mm"] = displacements["uy_mm"].round(6)
    #         st.dataframe(displacements, width="stretch")

    #     st.subheader("Member forces, mass and cost")

    #     members_out = results["members"].copy()
    #     members_out["length_m"] = members_out["length_m"].round(3)
    #     members_out["axial_force_N"] = members_out["axial_force_N"].round(3)
    #     members_out["stress_MPa"] = members_out["stress_MPa"].round(3)
    #     members_out["mass_kg"] = members_out["mass_kg"].round(3)
    #     members_out["cost_EUR"] = members_out["cost_EUR"].round(2)

    #     st.dataframe(members_out, width="stretch")

    #     total_mass = results["members"]["mass_kg"].sum()
    #     total_cost = results["members"]["cost_EUR"].sum()

    #     st.metric("Total material mass", f"{total_mass:.3f} kg")
    #     st.metric("Estimated material cost", f"{total_cost:.2f} €")

    # except Exception as e:
    #     st.error(str(e))