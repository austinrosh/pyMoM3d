"""Progress reporting for MoM simulation stages.

Provides a callback-based reporter abstraction so that solver internals
never import I/O code directly.  Two implementations are included:

* ``TerminalReporter`` – writes human-readable stage messages to a stream
  (default: *stderr*), with ``\\r``-based in-place progress updates when
  the stream is a TTY.
* ``SilentReporter`` – no-op, for tests and library embedding.
"""

import sys
import time
from typing import Any, Optional, TextIO


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string."""
    if seconds < 1.0:
        return f"{seconds:.1f} s"
    elif seconds < 60.0:
        return f"{seconds:.1f} s"
    elif seconds < 3600.0:
        m = int(seconds) // 60
        s = seconds - m * 60
        return f"{m}m {s:.0f}s"
    else:
        h = int(seconds) // 3600
        m = (int(seconds) % 3600) // 60
        return f"{h}h {m:02d}m"


def _format_eta(elapsed: float, fraction: float) -> str:
    """Linear ETA extrapolation; returns '' until >=5% complete."""
    if fraction < 0.05 or fraction <= 0.0:
        return ""
    remaining = elapsed * (1.0 - fraction) / fraction
    return f"ETA {_format_elapsed(remaining)}"


class TerminalReporter:
    """Write stage-by-stage progress to a terminal stream.

    Parameters
    ----------
    stream : TextIO, optional
        Output stream (default ``sys.stderr``).
    verbosity : str
        ``'normal'`` (default) – full progress lines.
        ``'quiet'`` – stage completion lines only (no mid-stage updates).
    """

    TAG_WIDTH = 12  # fixed-width for ``[Stage]`` tags

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        verbosity: str = "normal",
    ):
        self._stream = stream or sys.stderr
        self._verbosity = verbosity
        self._is_tty = hasattr(self._stream, "isatty") and self._stream.isatty()
        self._stage_starts: dict[str, float] = {}
        self._last_progress_time: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stage_start(self, name: str, **meta: Any) -> None:
        """Signal that stage *name* has begun."""
        self._stage_starts[name] = time.monotonic()
        self._last_progress_time[name] = 0.0
        tag = self._tag(name)
        msg = self._start_message(name, meta)
        self._write_line(f"{tag}{msg}")

    def stage_progress(self, name: str, fraction: float, **meta: Any) -> None:
        """Report mid-stage progress (0.0–1.0).  Throttled to ~2 s."""
        if self._verbosity == "quiet":
            return
        now = time.monotonic()
        if now - self._last_progress_time.get(name, 0.0) < 2.0:
            return
        self._last_progress_time[name] = now
        fraction = min(fraction, 0.999)  # never show 100% until stage_end
        elapsed = now - self._stage_starts.get(name, now)
        tag = self._tag(name)
        eta = _format_eta(elapsed, fraction)
        detail = self._progress_detail(name, fraction, meta)
        parts = [f"{tag}{fraction*100:5.1f}%"]
        if detail:
            parts.append(detail)
        if eta:
            parts.append(eta)
        line = "  |  ".join(parts)
        self._overwrite(line)

    def stage_end(self, name: str, **meta: Any) -> None:
        """Signal that stage *name* has finished."""
        elapsed = time.monotonic() - self._stage_starts.pop(name, time.monotonic())
        self._last_progress_time.pop(name, None)
        tag = self._tag(name)
        detail = self._end_message(name, meta, elapsed)
        self._finish_overwrite(f"{tag}Done: {detail} ({_format_elapsed(elapsed)})")

    def warning(self, msg: str) -> None:
        """Emit a warning line."""
        self._finish_overwrite(f"{'[WARNING]':<{self.TAG_WIDTH}}{msg}")

    def error(self, msg: str) -> None:
        """Emit an error line."""
        self._finish_overwrite(f"{'[ERROR]':<{self.TAG_WIDTH}}{msg}")

    def finish(self) -> None:
        """Ensure the terminal is left in a clean state (trailing newline)."""
        pass  # stage_end already writes newlines

    # ------------------------------------------------------------------
    # Internal formatting helpers
    # ------------------------------------------------------------------

    def _tag(self, name: str) -> str:
        label = f"[{_STAGE_LABELS.get(name, name)}]"
        return f"{label:<{self.TAG_WIDTH}}"

    @staticmethod
    def _start_message(name: str, meta: dict) -> str:
        if name == "mesh":
            geom = meta.get("geometry_type", "geometry")
            return f"Generating mesh for {geom}..."
        if name == "rwg":
            return "Computing basis functions..."
        if name == "z_fill":
            N = meta.get("N", "?")
            pairs = meta.get("total_pairs", "?")
            q = meta.get("quad_order", "?")
            return f"Assembling {N}x{N} matrix ({pairs} pairs, quad={q})..."
        if name == "solve_direct":
            N = meta.get("N", "?")
            return f"Direct LU, N={N}..."
        if name == "solve_gmres":
            N = meta.get("N", "?")
            tol = meta.get("tol", "?")
            return f"GMRES, N={N}, tol={tol}..."
        if name == "far_field":
            M = meta.get("num_angles", "?")
            return f"Computing {M} observation angles..."
        if name == "sweep":
            K = meta.get("num_freqs", "?")
            f_min = meta.get("f_min_ghz", "?")
            f_max = meta.get("f_max_ghz", "?")
            return f"{K} frequencies, {f_min}-{f_max} GHz"
        if name == "sweep_freq":
            idx = meta.get("index", "?")
            total = meta.get("total", "?")
            freq = meta.get("freq_ghz", "?")
            return f"f = {freq} GHz"
        return "..."

    @staticmethod
    def _progress_detail(name: str, fraction: float, meta: dict) -> str:
        if name == "z_fill":
            row = meta.get("row")
            N = meta.get("N")
            if row is not None and N is not None:
                return f"row {row}/{N}"
        if name == "solve_gmres":
            it = meta.get("iteration")
            residual = meta.get("residual")
            if it is not None:
                r_str = f", residual={residual:.2e}" if residual is not None else ""
                return f"iter {it}{r_str}"
        return ""

    @staticmethod
    def _end_message(name: str, meta: dict, elapsed: float) -> str:
        if name == "mesh":
            nt = meta.get("num_triangles", "?")
            nv = meta.get("num_vertices", "?")
            me = meta.get("mean_edge", None)
            me_str = f", mean edge {me:.4f} m" if me is not None else ""
            return f"{nt} triangles, {nv} vertices{me_str}"
        if name == "rwg":
            ni = meta.get("num_interior", "?")
            nb = meta.get("num_boundary", "?")
            return f"{ni} interior edges, {nb} boundary edges"
        if name == "z_fill":
            N = meta.get("N", "?")
            rate = meta.get("rate")
            rate_str = f" ({rate:.0f} pairs/s)" if rate is not None else ""
            return f"{N}x{N}{rate_str}"
        if name == "solve_direct":
            cond = meta.get("cond")
            res = meta.get("residual")
            parts = []
            if cond is not None:
                parts.append(f"cond={cond:.2e}")
            if res is not None:
                parts.append(f"residual={res:.2e}")
            return ", ".join(parts) if parts else "solved"
        if name == "solve_gmres":
            iters = meta.get("iterations")
            res = meta.get("residual")
            parts = []
            if iters is not None:
                parts.append(f"{iters} iters")
            if res is not None:
                parts.append(f"residual={res:.2e}")
            return ", ".join(parts) if parts else "solved"
        if name == "far_field":
            return f"{meta.get('num_angles', '?')} angles"
        if name == "sweep":
            K = meta.get("num_freqs", "?")
            return f"{K} frequencies"
        return ""

    # ------------------------------------------------------------------
    # Low-level output
    # ------------------------------------------------------------------

    def _write_line(self, text: str) -> None:
        self._stream.write(text + "\n")
        self._stream.flush()

    def _overwrite(self, text: str) -> None:
        """Overwrite the current line (TTY) or print a new line (pipe)."""
        if self._is_tty:
            self._stream.write("\r" + text + "  ")
            self._stream.flush()
        else:
            self._stream.write(text + "\n")
            self._stream.flush()

    def _finish_overwrite(self, text: str) -> None:
        """End an overwriting sequence with a final newline."""
        if self._is_tty:
            self._stream.write("\r" + text + "\n")
        else:
            self._stream.write(text + "\n")
        self._stream.flush()


class SilentReporter:
    """No-op reporter for tests and library embedding."""

    def stage_start(self, name: str, **meta: Any) -> None:
        pass

    def stage_progress(self, name: str, fraction: float, **meta: Any) -> None:
        pass

    def stage_end(self, name: str, **meta: Any) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass

    def finish(self) -> None:
        pass


class RecordingReporter:
    """Wraps an inner reporter, forwarding all calls while accumulating metadata.

    The accumulated metadata dict can be consumed by a report writer after
    the simulation completes.
    """

    def __init__(self, inner):
        self._inner = inner
        self._stage_starts: dict[str, float] = {}
        self.metadata: dict[str, Any] = {
            "warnings": [],
            "errors": [],
        }

    def stage_start(self, name: str, **meta: Any) -> None:
        self._stage_starts[name] = time.monotonic()
        self._record_start(name, meta)
        self._inner.stage_start(name, **meta)

    def stage_progress(self, name: str, fraction: float, **meta: Any) -> None:
        self._inner.stage_progress(name, fraction, **meta)

    def stage_end(self, name: str, **meta: Any) -> None:
        elapsed = time.monotonic() - self._stage_starts.pop(name, time.monotonic())
        self._record_end(name, meta, elapsed)
        self._inner.stage_end(name, **meta)

    def warning(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.metadata["warnings"].append(f"[{ts}] {msg}")
        self._inner.warning(msg)

    def error(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.metadata["errors"].append(f"[{ts}] {msg}")
        self._inner.error(msg)

    def finish(self) -> None:
        self._inner.finish()

    def _record_start(self, name: str, meta: dict) -> None:
        if name == "mesh":
            self.metadata.setdefault("mesh", {})
            self.metadata["mesh"]["geometry_type"] = meta.get("geometry_type")
        elif name == "z_fill":
            self.metadata.setdefault("z_fill", {})
            self.metadata["z_fill"]["N"] = meta.get("N")
            self.metadata["z_fill"]["total_pairs"] = meta.get("total_pairs")
        elif name == "solve_gmres":
            self.metadata.setdefault("solve", {})
            self.metadata["solve"]["type"] = "GMRES"
        elif name == "solve_direct":
            self.metadata.setdefault("solve", {})
            self.metadata["solve"]["type"] = "direct (LU)"

    def _record_end(self, name: str, meta: dict, elapsed: float) -> None:
        if name == "mesh":
            m = self.metadata.setdefault("mesh", {})
            m["num_triangles"] = meta.get("num_triangles")
            m["num_vertices"] = meta.get("num_vertices")
            m["mean_edge"] = meta.get("mean_edge")
            m["time"] = elapsed
        elif name == "rwg":
            r = self.metadata.setdefault("rwg", {})
            r["num_interior"] = meta.get("num_interior")
            r["num_boundary"] = meta.get("num_boundary")
            r["time"] = elapsed
        elif name == "z_fill":
            z = self.metadata.setdefault("z_fill", {})
            z["time"] = elapsed
            z["N"] = meta.get("N", z.get("N"))
        elif name == "solve_direct":
            s = self.metadata.setdefault("solve", {})
            s["cond"] = meta.get("cond")
            s["residual"] = meta.get("residual")
            s["time"] = elapsed
        elif name == "solve_gmres":
            s = self.metadata.setdefault("solve", {})
            s["iterations"] = meta.get("iterations")
            s["residual"] = meta.get("residual")
            s["time"] = elapsed


# Stage name -> short display label
_STAGE_LABELS = {
    "mesh": "Mesh",
    "rwg": "RWG",
    "z_fill": "Z-fill",
    "solve_direct": "Solve",
    "solve_gmres": "Solve",
    "far_field": "Far-field",
    "sweep": "Sweep",
    "sweep_freq": "Sweep",
}
