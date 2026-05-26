"""
1D stellar-interior model for a Sun-like star.

Solves the four stellar structure ODEs from the surface inward using adaptive
RK4 with step-doubling error control.  Opacity and energy generation rates are
interpolated from tabulated data (bilinear by default).
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq

# Physical constants (SI)
G       = 6.6743e-11          # m^3 kg^-1 s^-2
sigma   = 5.670374419e-8      # W m^-2 K^-4
c_light = 2.99792458e8        # m s^-1
k_B     = 1.380649e-23        # J K^-1
m_u     = 1.66053906660e-27   # kg


# Interpolation 
class TableInterpolator:
    """Bilinear interpolation of an opacity or epsilon table in log-space."""

    def __init__(self, path, si_factor, name="", kx=1, ky=1):
        self.name = name or path
        self.si_factor = si_factor

        with open(path) as f:
            self.log_R = np.array(f.readline().split()[1:], dtype=float)

        data = np.loadtxt(path, skiprows=1)
        self.log_T = data[:, 0]
        self._spline = RectBivariateSpline(self.log_T, self.log_R, data[:, 1:], kx=kx, ky=ky)

    def __call__(self, T, rho, warn=True):
        """Return interpolated quantity in SI from T [K] and rho [kg/m^3]."""
        rho_cgs = rho * 1e-3
        logT = np.log10(T)
        logR = np.log10(rho_cgs / (T / 1e6) ** 3)

        # Check bounds of interpolation
        if warn and (logT < self.log_T[0] or logT > self.log_T[-1] or logR < self.log_R[0] or logR > self.log_R[-1]):
            warnings.warn(f"{self.name} extrapolating: logT={logT:.3f}, logR={logR:.3f}")
        
        return 10 ** float(self._spline(logT, logR, grid=False)) * self.si_factor

    def check(self, sanity_rows, tol=0.05):
        """Run sanity check against known values. Returns True if all pass."""
        print(f"{'logT':>6} {'logR':>6} {'expected SI':>14} {'computed SI':>14} {'rel.err':>9}")
        failed = 0
        for logT, logR, expected in sanity_rows:
            T = 10 ** logT
            computed = self(T, 10 ** logR * (T / 1e6) ** 3 * 1e3, warn=False)
            err = abs(computed - expected) / expected
            if err >= tol: failed += 1
            print(f"{logT:6.3f} {logR:6.2f} {expected:14.3e} "f"{computed:14.3e} {err * 100:8.2f}%{'  FAIL' if err >= tol else ''}")
        ok = failed == 0
        tag = "passed" if ok else f"WARNING: {failed} rows failed (>{tol*100:.0f}%)"
        print(f"{self.name}: {tag}")
        return ok


# Stellar model
class StellarModel:
    """1D stellar interior model: EOS, ODE integration, and plotting."""

    # I'd like to keep track of why we're stopping
    clean_stops = {"reached zero", "max_steps", "mass exhausted", "centre reached"}
    def __init__(self, X, Y, kappa, epsilon, M0, R0, L0, T0, rho0=None, P0=None):
        if rho0 is None and P0 is None:
            raise ValueError("Provide either rho0 or P0")

        # This is from the assignment
        self.X, self.Y = X, Y
        self.Z = 1.0 - X - Y
        self.mu = 1.0 / (2 * X + 3 * Y / 4 + self.Z / 2)
        self.Cp = 5 * k_B / (2 * self.mu * m_u)
        self._a3c = 4 * sigma / (3 * c_light)
        self.kappa, self.epsilon = kappa, epsilon

        self.M0, self.R0, self.L0, self.T0 = M0, R0, L0, T0
        self.P0 = P0 if P0 is not None else self.pressure(rho0, T0)
        self.result = None
        self.stop_reason = ""

    #  Equation of state
    def pressure(self, rho, T):
        """Total pressure P [Pa] = radiation + ideal gas."""
        return self._a3c * T ** 4 + rho * k_B * T / (self.mu * m_u)

    def density(self, P, T):
        """Density [kg/m^3] from pressure and temperature."""
        return (P - self._a3c * T ** 4) * self.mu * m_u / (k_B * T)

    #  Temperature gradients
    def compute_gradients(self, m, r, T, P, rho, L, kappa_val):
        """Return (nabla_stable, nabla_ad, nabla_star, is_convective)."""
        g  = G * m / r ** 2
        Hp = k_B * T / (self.mu * m_u * g)
        nabla_stable = 3 * kappa_val * rho * Hp * L / (64 * np.pi * r**2 * sigma * T**4)
        nabla_ad = P / (T * rho * self.Cp)

        if nabla_stable <= nabla_ad:
            return nabla_stable, nabla_ad, nabla_stable, False

        # Convective: solve mixing-length cubic for xi
        lm = Hp
        U  = (64 * sigma * T**3) / (3 * kappa_val * rho**2 * self.Cp) * np.sqrt(Hp / g)
        Om = 4.0 / lm
        a2, a1 = U / lm**2, (U**2) * Om / lm**3
        a0 = (U / lm**2) * (nabla_ad - nabla_stable)

        f  = lambda xi: ((xi + a2) * xi + a1) * xi + a0
        hi = 1.0
        for _ in range(200):
            if f(hi) > 0: break
            hi *= 2.0
        else:
            raise RuntimeError("xi root not bracketed")

        xi = brentq(f, 0.0, hi, xtol=1e-12, rtol=1e-10)
        return nabla_stable, nabla_ad, xi**2 + (U * Om / lm) * xi + nabla_ad, True

    # ODE system to solve
    def _rhs(self, m, y):
        """dy/dm for the four stellar structure equations. y = [r, P, L, T]."""
        r, P, L, T = y
        rho = self.density(P, T)
        kv  = self.kappa(T, rho, warn=False)
        _, _, nstar, is_conv = self.compute_gradients(m, r, T, P, rho, L, kv)
        dP = -G * m / (4 * np.pi * r**4)
        dT = (nstar * (T / P) * dP if is_conv
              else -3 * kv * L / (256 * np.pi**2 * sigma * r**4 * T**3))
        return np.array([1.0 / (4 * np.pi * r**2 * rho), dP,
                         self.epsilon(T, rho, warn=False), dT])

    def _diagnostics(self, m, y):
        """Derived quantities at one mass shell."""
        r, P, L, T = y
        rho = self.density(P, T)
        kv  = self.kappa(T, rho, warn=False)
        ns, nad, nstar, conv = self.compute_gradients(m, r, T, P, rho, L, kv)
        return dict(rho=rho, kappa=kv, eps=self.epsilon(T, rho, warn=False),
                    nstable=ns, nad=nad, nstar=nstar, is_conv=conv)

    # Adaptive RK4 integrator 
    def _rk4_step(self, m, y, dm):
        k1 = self._rhs(m,        y)
        k2 = self._rhs(m + dm/2, y + dm * k1 / 2)
        k3 = self._rhs(m + dm/2, y + dm * k2 / 2)
        k4 = self._rhs(m + dm,   y + dm * k3)
        return y + dm * (k1 + 2*k2 + 2*k3 + k4) / 6

    def _adaptive_step(self, m, y, dm, eps_want):
        """One accepted step via step-doubling. Returns (m_new, y_new, dm_next)."""
        floor = abs(dm) * 1e-12
        for _ in range(25):
            try:
                y_big   = self._rk4_step(m, y, dm)
                y_h1    = self._rk4_step(m, y, dm / 2)
                y_small = self._rk4_step(m + dm / 2, y_h1, dm / 2)
                ok = all(np.all(np.isfinite(a)) and a[0] > 0 and a[1] > 0 and a[3] > 0
                         for a in (y_big, y_h1, y_small))
            except Exception:
                ok = False
            if not ok:
                dm *= 0.25
                if abs(dm) < floor: raise RuntimeError(f"dm below floor ({dm:.2e})")
                continue
            eps_rel = float(np.max(np.abs(y_small - y_big) / np.maximum(np.abs(y_small), 1e-30)))
            if eps_rel < 1e-30:
                return m + dm, y_small, dm * 5.0
            ratio = np.clip(0.9 * (eps_want / eps_rel) ** 0.2, 0.1, 5.0)
            if eps_rel <= eps_want:
                return m + dm, y_small, dm * ratio
            dm *= max(0.1, ratio)
            if abs(dm) < floor: raise RuntimeError(f"dm below floor ({dm:.2e})")
        raise RuntimeError(f"too many rejections (dm={dm:.2e})")

    def run(self, eps_want=1e-3, max_steps=30000, verbose=False):
        """Integrate from surface to centre. Stores result in self.result."""
        y  = np.array([self.R0, self.P0, self.L0, self.T0])
        m, r0_init, dm = float(self.M0), float(self.R0), -self.M0 * 1e-5

        h = {k: [v] for k, v in zip("rPLT", y)}
        h["m"] = [m]
        for k, v in self._diagnostics(m, y).items(): h[k] = [v]
        stop = "max_steps"

        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(max_steps):
                if m + dm < 0: dm = -m
                try:
                    m, y, dm = self._adaptive_step(m, y, dm, eps_want)
                except RuntimeError as e:
                    stop = f"step rejected: {e}"; break
                if y[1] - self._a3c * y[3] ** 4 <= 0:
                    stop = "P drops below radiation pressure"; break
                h["m"].append(m)
                for k, v in zip("rPLT", y): h[k].append(v)
                for k, v in self._diagnostics(m, y).items(): h[k].append(v)
                if m <= 0 or y[0] <= 0 or y[2] <= 0:   stop = "reached zero"; break
                if abs(m) < abs(self.M0) * 1e-8:        stop = "mass exhausted"; break
                if abs(y[0]) < r0_init * 1e-4:          stop = "centre reached"; break

        self.result = {k: np.array(v) for k, v in h.items()}
        self.result["stop_reason"] = stop
        self.stop_reason = stop
        if verbose: print(f"stopped after {len(self.result['m']) - 1} steps: {stop}")
        return self

    # Goal analysis 
    def _zone_bounds(self, r, L, is_conv):
        """Return (core_outer_r, conv_inner_r, conv_outer_r)."""
        L_norm = L / L[0]
        core_mask = L_norm < 0.995
        core_r = r[core_mask].max() if core_mask.any() else 0.0
        if not is_conv.any():
            return core_r, 0.0, 0.0
        first = int(np.argmax(is_conv))
        rest  = is_conv[first:]
        end   = (first + int(np.argmax(~rest))) if (~rest).any() else len(is_conv)
        return core_r, r[end - 1], r[first]

    def goal_summary(self):
        """Return dict of goal metrics: m_frac, r_frac, L_frac, core_outer, conv_width, stop."""
        res = self.result
        core_r, conv_in, conv_out = self._zone_bounds(res["r"], res["L"], res["is_conv"])
        conv_width = (conv_out - conv_in) / self.R0 if conv_out > conv_in else 0.0
        return dict(m_frac=res["m"][-1] / self.M0, r_frac=res["r"][-1] / self.R0,
                    L_frac=res["L"][-1] / self.L0, core_outer=core_r / self.R0,
                    conv_width=conv_width, stop=res["stop_reason"])

    # Plotting 
    # Zone colour scheme (shared by profiles + cross-section)
    ZONE_COLOURS = {"conv_out": "#d62728", "rad_out": "#ffd23f",
                     "rad_in":  "#7fd0e3", "conv_in": "#1f3b8a"}

    def _trim_result(self):
        """Return (trimmed dict, normalised r) cutting the singularity tail."""
        res  = self.result
        keep = (res["T"] < 3e7) & (res["r"] / res["r"][0] >= 1e-20)
        trimmed = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in res.items()}
        return trimmed, trimmed["r"] / self.R0

    def plot_profiles(self, savepath=None):
        """Six-panel radial profiles with zone shading."""
        res, r = self._trim_result()
        core_r, conv_in, conv_out = [v / self.R0 for v in
            self._zone_bounds(res["r"], res["L"], res["is_conv"])]

        # Inner convection boundary (within core)
        in_core = r <= core_r if core_r > 0 else np.zeros(len(r), dtype=bool)
        core_conv = in_core & res["is_conv"]
        inner_conv_outer = float(r[core_conv].max()) if core_conv.any() else 0.0

        C = self.ZONE_COLOURS
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        panels = [
            ("(a) Mass",        res["m"] / res["m"][0], r"$m / M_0$",                   "linear"),
            ("(b) Luminosity",  res["L"] / res["L"][0], r"$L / L_0$",                   "linear"),
            ("(c) Temperature", res["T"],               r"$T$ [K]",                     "log"),
            ("(d) Pressure",    res["P"],               r"$P$ [Pa]",                    "log"),
            ("(e) Density",     res["rho"],             r"$\rho$ [kg m$^{-3}$]",        "log"),
            ("(f) Energy gen.", res["eps"],             r"$\varepsilon$ [W kg$^{-1}$]", "log"),
        ]
        for ax, (title, y, ylabel, scale) in zip(axes.flat, panels):
            # Zone shading (surface inward)
            if conv_out < 0.99:
                ax.axvspan(conv_out, 1.0, color=C["rad_out"], alpha=0.10, lw=0)
            if conv_out > conv_in > 0:
                ax.axvspan(conv_in, conv_out, color=C["conv_out"], alpha=0.10, lw=0)
            if conv_in > core_r:
                ax.axvspan(core_r, conv_in, color=C["rad_out"], alpha=0.10, lw=0)
            elif conv_out == 0 and core_r < 1.0:
                ax.axvspan(core_r, 1.0, color=C["rad_out"], alpha=0.10, lw=0)
            if inner_conv_outer > 0:
                ax.axvspan(inner_conv_outer, core_r, color=C["rad_in"], alpha=0.10, lw=0)
                ax.axvspan(0, inner_conv_outer, color=C["conv_in"], alpha=0.10, lw=0)
            elif core_r > 0:
                ax.axvspan(0, core_r, color=C["rad_in"], alpha=0.10, lw=0)

            ax.plot(r, y, color="k", lw=1.8)
            ax.set_title(title, fontsize=13, loc="left")
            ax.set_xlabel(r"$r / R_0$"); ax.set_ylabel(ylabel); ax.set_xlim(0, 1.0)
            if scale == "log":
                ax.set_yscale("log")
                yp = y[(y > 0) & np.isfinite(y)]
                if yp.size:
                    yhi = float(np.percentile(yp, 95))
                    ylo = float(max(yp.min(), yhi * 1e-10))
                    ax.set_ylim(ylo / 3, yhi * 10)

        fig.legend(handles=[
            Patch(fc=C["conv_out"], alpha=0.35, label="Conv. (envelope)"),
            Patch(fc=C["rad_out"],  alpha=0.35, label="Rad. (envelope)"),
            Patch(fc=C["rad_in"],   alpha=0.35, label="Rad. (core)"),
            Patch(fc=C["conv_in"],  alpha=0.35, label="Conv. (core)"),
        ], loc="lower center", ncol=4, fontsize=9, frameon=False,
            bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        if savepath: fig.savefig(savepath)
        return fig

    def plot_gradients(self, savepath=None):
        """Temperature gradients vs r/R0."""
        res, r = self._trim_result()
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.semilogy(r, res["nstable"], lw=2, color="#1f77b4", label=r"$\nabla_{\rm stable}$")
        ax.semilogy(r, res["nstar"],   lw=2, color="#ff7f0e", label=r"$\nabla^{*}$")
        ax.semilogy(r, res["nad"],     lw=2, color="#2ca02c", ls="--", label=r"$\nabla_{\rm ad}$")
        ax.set_ylim(1e-1, 1e2); ax.set_xlim(0, 1.0)
        ax.set_xlabel(r"$r / R_0$"); ax.set_ylabel(r"$\nabla$")
        ax.set_title("Temperature gradients", loc="left", fontsize=13)
        ax.legend(loc="upper center", ncol=2)
        fig.tight_layout()
        if savepath: fig.savefig(savepath)
        return fig

    def draw_cross_section(self, ax, title="Cross section of star"):
        """Draw the cross-section on a given axes."""
        res, r = self._trim_result()
        L, L0, is_conv = res["L"], res["L"][0], res["is_conv"]
        core_r, conv_in, conv_out = [v / self.R0 for v in
            self._zone_bounds(res["r"], L, is_conv)]

        colours = {(True, True): "#1f3b8a", (True, False): "#7fd0e3",
                   (False, True): "#d62728", (False, False): "#ffd23f"}
        order = np.argsort(-r)
        prev_col, groups = None, []
        for i in order:
            col = colours[(L[i] < 0.995 * L0, bool(is_conv[i]))]
            if col != prev_col: groups.append((r[i], col)); prev_col = col
        for ri, col in groups:
            ax.add_patch(Circle((0, 0), ri, fc=col, ec=col, lw=0))

        r_inner = float(r.min())
        if r_inner > 0.005:
            ax.add_patch(Circle((0, 0), r_inner, fc="white", ec="gray",
                                lw=1.0, ls="--", zorder=5))

        lim = float(np.max(r)) * 1.10
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel(r"$r / R_0$"); ax.set_ylabel(r"$r / R_0$")
        ax.set_title(title, loc="left", fontsize=13)
        info = []
        if conv_out > conv_in:
            info.append(f"outer conv.:  {conv_in:.2f}–{conv_out:.2f} $R_0$  (width {conv_out - conv_in:.2f})")
        if core_r > 0:
            info.append(f"core radius:  {core_r:.2f} $R_0$")
        if info:
            ax.text(0.02, 0.02, "\n".join(info), transform=ax.transAxes, fontsize=9,
                    va="bottom", bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                           ec="lightgray", alpha=0.9))

    def plot_cross_section(self, savepath=None):
        """Standalone cross-section figure."""
        fig, ax = plt.subplots(figsize=(8, 8))
        self.draw_cross_section(ax)
        ax.legend(handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=14, label=l)
            for c, l in [("#d62728", "Convection outside core"), ("#ffd23f", "Radiation outside core"),
                         ("#7fd0e3", "Radiation inside core"),   ("#1f3b8a", "Convection inside core")]
        ], loc="upper right")
        fig.tight_layout()
        if savepath: fig.savefig(savepath)
        return fig

    def plot_all(self, prefix="fig"):
        """Generate all three plot types."""
        self.plot_profiles(savepath=f"{prefix}_profiles.png");           plt.show()
        self.plot_gradients(savepath=f"{prefix}_gradients.png");         plt.show()
        self.plot_cross_section(savepath=f"{prefix}_cross_section.png"); plt.show()
