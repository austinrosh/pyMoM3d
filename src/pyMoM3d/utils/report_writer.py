"""Format accumulated simulation metadata into a plain-text report file."""

import os
from datetime import datetime, timezone


def _fmt_time(seconds):
    """Format seconds into human-readable string."""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.2f} s"
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}m {s:.0f}s"


def _fmt_float(val, fmt=".4g"):
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def _fmt_int(val):
    if val is None:
        return "N/A"
    return f"{val:,}"


def write_report(metadata, path):
    """Write a simulation report from accumulated metadata.

    Parameters
    ----------
    metadata : dict
        Structured metadata accumulated by RecordingReporter.
    path : str
        Output file path.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lines = _build_report(metadata)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        # Report writing failures should not crash the simulation
        pass


def _build_report(md):
    """Build report lines from metadata dict."""
    sep = "=" * 65
    lines = []

    # Header
    lines.append(sep)
    lines.append("pyMoM3d Simulation Report")
    lines.append(sep)
    lines.append(f"Generated:  {md.get('timestamp', 'N/A')}")
    lines.append(f"Run ID:     {md.get('run_id', 'N/A')}")
    lines.append(f"Status:     {md.get('status', 'UNKNOWN')}")
    lines.append(f"Runtime:    {_fmt_time(md.get('total_time'))}")
    lines.append(f"Version:    {md.get('version', 'N/A')}")
    lines.append(sep)

    # Configuration
    cfg = md.get("config", {})
    freq = cfg.get("frequency")
    wl = cfg.get("wavelength")
    lines.append("")
    lines.append("CONFIGURATION")
    lines.append("-------------")
    if freq is not None:
        freq_ghz = freq / 1e9
        wl_str = f" (λ = {wl:.3f} m)" if wl else ""
        lines.append(f"Frequency:          {freq_ghz:.3f} GHz{wl_str}")
    lines.append(f"Excitation:         {cfg.get('excitation', 'N/A')}")
    lines.append(f"Solver:             {cfg.get('solver_type', 'N/A')}")
    lines.append(f"Quadrature order:   {cfg.get('quad_order', 'N/A')}")
    lines.append(f"Near threshold:     {cfg.get('near_threshold', 'N/A')}")

    # Geometry & Mesh
    mesh = md.get("mesh", {})
    if mesh:
        lines.append("")
        lines.append("GEOMETRY & MESH")
        lines.append("---------------")
        lines.append(f"Geometry:           {mesh.get('geometry_type', 'N/A')}")
        lines.append(f"Mesher:             {mesh.get('mesher', 'N/A')}")
        tel = mesh.get("target_edge_length")
        if tel is not None:
            lines.append(f"Target edge length: {tel:.2e} m")
        lines.append(f"Triangles:          {_fmt_int(mesh.get('num_triangles'))}")
        lines.append(f"Vertices:           {_fmt_int(mesh.get('num_vertices'))}")
        me = mesh.get("mean_edge")
        if me is not None and wl is not None and wl > 0:
            elems_per_wl = wl / me
            lines.append(f"Mean edge length:   {me:.2e} m (λ/{elems_per_wl:.1f})")
        elif me is not None:
            lines.append(f"Mean edge length:   {me:.2e} m")
        mt = mesh.get("time")
        if mt is not None:
            lines.append(f"Mesh time:          {_fmt_time(mt)}")

    # RWG
    rwg = md.get("rwg", {})
    if rwg:
        lines.append("")
        lines.append("RWG BASIS")
        lines.append("---------")
        lines.append(f"Interior edges:     {_fmt_int(rwg.get('num_interior'))}")
        lines.append(f"Boundary edges:     {_fmt_int(rwg.get('num_boundary'))}")
        rt = rwg.get("time")
        if rt is not None:
            lines.append(f"Computation time:   {_fmt_time(rt)}")

    # Matrix assembly
    zfill = md.get("z_fill", {})
    if zfill:
        lines.append("")
        lines.append("MATRIX ASSEMBLY")
        lines.append("---------------")
        N = zfill.get("N")
        if N is not None:
            lines.append(f"Size:               {N} x {N}")
        lines.append(f"Pairs:              {_fmt_int(zfill.get('total_pairs'))}")
        ft = zfill.get("time")
        if ft is not None:
            lines.append(f"Fill time:          {_fmt_time(ft)}")
            pairs = zfill.get("total_pairs")
            if pairs and ft > 0:
                lines.append(f"Fill rate:          {_fmt_int(int(pairs / ft))} pairs/s")
        zmem = zfill.get("z_memory_mb")
        if zmem is not None:
            lines.append(f"Z memory:           {zmem:.2f} MB")

    # Solve
    solve = md.get("solve", {})
    if solve:
        lines.append("")
        lines.append("LINEAR SOLVE")
        lines.append("------------")
        lines.append(f"Solver:             {solve.get('type', 'N/A')}")
        cond = solve.get("cond")
        if cond is not None:
            lines.append(f"Condition number:   {cond:.2e}")
        res = solve.get("residual")
        if res is not None:
            lines.append(f"Residual:           {res:.2e}")
        st = solve.get("time")
        if st is not None:
            lines.append(f"Solve time:         {_fmt_time(st)}")
        iters = solve.get("iterations")
        if iters is not None:
            lines.append(f"GMRES iterations:   {iters}")
        conv = solve.get("converged")
        if conv is not None:
            lines.append(f"Converged:          {conv}")

    # Results
    results = md.get("results", {})
    if results:
        lines.append("")
        lines.append("RESULTS")
        lines.append("-------")
        zin = results.get("Z_input")
        if zin is not None:
            lines.append(f"Input impedance:    {zin.real:.2f} + j{zin.imag:.2f} Ω")

    # Warnings
    warnings = md.get("warnings", [])
    if warnings:
        lines.append("")
        lines.append(f"WARNINGS ({len(warnings)})")
        lines.append("-" * max(12, len(f"WARNINGS ({len(warnings)})")))
        for w in warnings:
            lines.append(w)

    # Errors
    errors = md.get("errors", [])
    if errors:
        lines.append("")
        lines.append(f"ERRORS ({len(errors)})")
        lines.append("-" * max(10, len(f"ERRORS ({len(errors)})")))
        for e in errors:
            lines.append(e)

    lines.append("")
    lines.append(sep)
    lines.append("End of report")
    lines.append(sep)

    return lines
