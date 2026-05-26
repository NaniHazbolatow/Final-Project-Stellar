import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq

# Physical constants (SI units)
G = 6.6743e-11
sigma = 5.670374e-8
c_light = 2.99792458e8
k_B = 1.380649e-23
m_u = 1.660539e-27


class TableInterpolator:
    """Bilinear interpolation for opacity/epsilon tables in log-space."""
    def __init__(self, path, si_factor, name="", kx=1, ky=1):
        self.name = name or path
        self.si_factor = si_factor

        with open(path) as f:
            self.log_R = np.array(f.readline().split()[1:], dtype=float)

        data = np.loadtxt(path, skiprows=1)
        self.log_T = data[:, 0]
        self._spline = RectBivariateSpline(self.log_T, self.log_R, data[:, 1:], kx=kx, ky=ky)

    def __call__(self, T, rho, warn=True):
        rho_cgs = rho * 1e-3
        logT = np.log10(T)
        logR = np.log10(rho_cgs / (T / 1e6) ** 3)

        # Bounds check
        if warn and (logT < self.log_T[0] or logT > self.log_T[-1] or logR < self.log_R[0] or logR > self.log_R[-1]):
            warnings.warn(f"{self.name} extrapolating: logT={logT:.3f}, logR={logR:.3f}")
        
        return 10 ** float(self._spline(logT, logR, grid=False)) * self.si_factor

    def check(self, sanity_rows, tol=0.05):
        """Sanity check against validation values."""
        print(f"{'logT':>6} {'logR':>6} {'expected':>12} {'computed':>12} {'err %':>8}")
        failed = 0
        for logT, logR, expected in sanity_rows:
            T = 10 ** logT
            rho = 10 ** logR * (T / 1e6) ** 3 * 1e3
            computed = self(T, rho, warn=False)
            err = abs(computed - expected) / expected
            if err >= tol: 
                failed += 1
            print(f"{logT:6.3f} {logR:6.2f} {expected:12.3e} {computed:12.3e} {err * 100:7.2f}%{'  FAIL' if err >= tol else ''}")
        
        if failed == 0:
            print(f"{self.name}: passed sanity check.")
        else:
            print(f"{self.name}: WARNING - {failed} rows failed!")
        return failed == 0


class StellarModel:
    def __init__(self, X, Y, kappa, epsilon, M0, R0, L0, T0, rho0=None, P0=None):
        if rho0 is None and P0 is None:
            raise ValueError("Must provide either rho0 or P0")

        self.X = X
        self.Y = Y
        self.Z = 1.0 - X - Y
        self.mu = 1.0 / (2 * X + 0.75 * Y + 0.5 * self.Z)
        self.Cp = 2.5 * k_B / (self.mu * m_u)
        self._a3c = 4 * sigma / (3 * c_light)
        self.kappa = kappa
        self.epsilon = epsilon

        self.M0 = M0
        self.R0 = R0
        self.L0 = L0
        self.T0 = T0
        self.P0 = P0 if P0 is not None else self.pressure(rho0, T0)
        self.result = None
        self.stop_reason = ""

    def pressure(self, rho, T):
        # Radiation + Ideal Gas
        return self._a3c * T**4 + rho * k_B * T / (self.mu * m_u)

    def density(self, P, T):
        return (P - self._a3c * T**4) * self.mu * m_u / (k_B * T)

def compute_gradients(self, m, r, T, P, rho, L, kappa_val):
        g = G * m / r**2
        Hp = k_B * T / (self.mu * m_u * g)
        nabla_stable = 3 * kappa_val * rho * Hp * L / (64 * np.pi * r**2 * sigma * T**4)
        nabla_ad = P / (T * rho * self.Cp)

        if nabla_stable <= nabla_ad:
            return nabla_stable, nabla_ad, nabla_stable, False

        # Mixing-length cubic solver for convective zones
        lm = Hp
        U = (64 * sigma * T**3) / (3 * kappa_val * rho**2 * self.Cp) * np.sqrt(Hp / g)
        Om = 4.0 / lm
        a2 = U / lm**2
        a1 = (U**2) * Om / lm**3
        a0 = (U / lm**2) * (nabla_ad - nabla_stable)

        f = lambda xi: ((xi + a2) * xi + a1) * xi + a0
        
        # Robust root bracketing
        hi = max(1.0, abs(a0)**0.33)
        while f(hi) <= 0:
            hi *= 10.0

        xi = brentq(f, 0.0, hi, xtol=1e-12)
        nabla_star = xi**2 + (U * Om / lm) * xi + nabla_ad
        return nabla_stable, nabla_ad, nabla_star, True

    def _rhs(self, m, y):
        """Stellar structure ODEs: y = [r, P, L, T]"""
        r, P, L, T = y
        rho = self.density(P, T)
        kv = self.kappa(T, rho, warn=False)
        _, _, nstar, is_conv = self.compute_gradients(m, r, T, P, rho, L, kv)
        
        dP = -G * m / (4 * np.pi * r**4)
        if is_conv:
            dT = nstar * (T / P) * dP
        else:
            dT = -3 * kv * L / (256 * np.pi**2 * sigma * r**4 * T**3)
            
        dr = 1.0 / (4 * np.pi * r**2 * rho)
        dL = self.epsilon(T, rho, warn=False)
        return np.array([dr, dP, dL, dT])

    def _diagnostics(self, m, y):
        r, P, L, T = y
        rho = self.density(P, T)
        kv = self.kappa(T, rho, warn=False)
        ns, nad, nstar, conv = self.compute_gradients(m, r, T, P, rho, L, kv)
        return {
            "rho": rho, "kappa": kv, "eps": self.epsilon(T, rho, warn=False),
            "nstable": ns, "nad": nad, "nstar": nstar, "is_conv": conv
        }

    def _rk4_step(self, m, y, dm):
        k1 = self._rhs(m, y)
        k2 = self._rhs(m + dm/2, y + dm * k1 / 2)
        k3 = self._rhs(m + dm/2, y + dm * k2 / 2)
        k4 = self._rhs(m + dm, y + dm * k3)
        return y + dm * (k1 + 2*k2 + 2*k3 + k4) / 6

    def _adaptive_step(self, m, y, dm, eps_want):
        """Step-doubling error control loop."""
        for _ in range(20):
            y_big = self._rk4_step(m, y, dm)
            y_h1 = self._rk4_step(m, y, dm / 2)
            y_small = self._rk4_step(m + dm / 2, y_h1, dm / 2)
            
            ok = all(np.all(np.isfinite(a)) and a[0] > 0 and a[1] > 0 and a[3] > 0 for a in (y_big, y_h1, y_small))
            if not ok:
                dm *= 0.5
                continue
                
            eps_rel = float(np.max(np.abs(y_small - y_big) / np.maximum(np.abs(y_small), 1e-30)))
            if eps_rel < 1e-20:
                return m + dm, y_small, dm * 2.0
                
            ratio = 0.9 * (eps_want / eps_rel) ** 0.2
            ratio = np.clip(ratio, 0.1, 5.0)
            
            if eps_rel <= eps_want:
                return m + dm, y_small, dm * ratio
            else:
                dm *= ratio
                
        raise RuntimeError("Adaptive step sizing failed to converge.")

    def run(self, eps_want=1e-3, max_steps=30000, verbose=False):
        y = np.array([self.R0, self.P0, self.L0, self.T0])
        m = float(self.M0)
        dm = -self.M0 * 1e-5

        # Initialize history tracking
        h = {k: [v] for k, v in zip("rPLT", y)}
        h["m"] = [m]
        for k, v in self._diagnostics(m, y).items(): 
            h[k] = [v]
            
        stop = "max_steps"

        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(max_steps):
                if m + dm < 0: 
                    dm = -m
                    
                m, y, dm = self._adaptive_step(m, y, dm, eps_want)
                
                if y[1] - self._a3c * y[3] ** 4 <= 0:
                    stop = "Gas pressure hit zero"
                    break
                    
                h["m"].append(m)
                for k, v in zip("rPLT", y): h[k].append(v)
                for k, v in self._diagnostics(m, y).items(): h[k].append(v)
                
                if m <= 0 or y[0] <= 0 or y[2] <= 0:   
                    stop = "reached zero"
                    break
                if abs(m) < self.M0 * 1e-6:        
                    stop = "mass exhausted"
                    break
                if abs(y[0]) < self.R0 * 1e-4:          
                    stop = "centre reached"
                    break

        self.result = {k: np.array(v) for k, v in h.items()}
        self.stop_reason = stop
        if verbose: 
            print(f"Stopped after {len(self.result['m'])} steps: {stop}")
        return self

    def _zone_bounds(self, r, L, is_conv):
        L_norm = L / L[0]
        core_mask = L_norm < 0.995
        core_r = r[core_mask].max() if core_mask.any() else 0.0
        if not is_conv.any():
            return core_r, 0.0, 0.0
        first = int(np.argmax(is_conv))
        rest = is_conv[first:]
        end = (first + int(np.argmax(~rest))) if (~rest).any() else len(is_conv)
        return core_r, r[end - 1], r[first]

    def goal_summary(self):
        res = self.result
        core_r, conv_in, conv_out = self._zone_bounds(res["r"], res["L"], res["is_conv"])
        conv_width = (conv_out - conv_in) / self.R0 if conv_out > conv_in else 0.0
        return {
            "m_frac": res["m"][-1] / self.M0,
            "r_frac": res["r"][-1] / self.R0,
            "L_frac": res["L"][-1] / self.L0,
            "core_outer": core_r / self.R0,
            "conv_width": conv_width,
            "stop": self.stop_reason
        }

    # Plotting styles
    ZONE_COLORS = {"conv_out": "#d62728", "rad_out": "#ffd23f", "rad_in": "#7fd0e3", "conv_in": "#1f3b8a"}

    def _trim_result(self):
        res = self.result
        keep = (res["T"] < 3e7) & (res["r"] / self.R0 >= 1e-5)
        return {k: v[keep] for k, v in res.items()}, res["r"][keep] / self.R0

    def plot_profiles(self, savepath=None):
        res, r = self._trim_result()
        core_r, conv_in, conv_out = [v / self.R0 for v in self._zone_bounds(res["r"], res["L"], res["is_conv"])]

        # Setup standard 2x3 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        C = self.ZONE_COLORS

        # Data map to cleanly unpack into loop
        plots_config = [
            (axes[0,0], "Mass", res["m"] / self.M0, r"$m / M_0$", False),
            (axes[0,1], "Luminosity", res["L"] / self.L0, r"$L / L_0$", False),
            (axes[0,2], "Temperature", res["T"], r"$T$ [K]", True),
            (axes[1,0], "Pressure", res["P"], r"$P$ [Pa]", True),
            (axes[1,1], "Density", res["rho"], r"$\rho$ [kg/m$^3$]", True),
            (axes[1,2], "Energy Generation", res["eps"], r"$\varepsilon$ [W/kg]", True),
        ]

        for ax, title, y_data, ylabel, is_log in plots_config:
            # Draw backgrounds for zones
            if conv_out < 0.99: ax.axvspan(conv_out, 1.0, color=C["rad_out"], alpha=0.1)
            if conv_out > conv_in > 0: ax.axvspan(conv_in, conv_out, color=C["conv_out"], alpha=0.1)
            if conv_in > core_r: ax.axvspan(core_r, conv_in, color=C["rad_out"], alpha=0.1)
            if core_r > 0: ax.axvspan(0, core_r, color=C["rad_in"], alpha=0.1)

            ax.plot(r, y_data, color="black", lw=1.5)
            ax.set_title(title, loc="left")
            ax.set_xlabel(r"$r / R_0$")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, 1.0)
            if is_log:
                ax.set_yscale("log")

        fig.legend(handles=[
            Patch(fc=C["conv_out"], alpha=0.4, label="Convective Envelope"),
            Patch(fc=C["rad_out"], alpha=0.4, label="Radiative Envelope"),
            Patch(fc=C["rad_in"], alpha=0.4, label="Core Zone"),
        ], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        if savepath: fig.savefig(savepath)
        return fig

    def plot_gradients(self, savepath=None):
        res, r = self._trim_result()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(r, res["nstable"], label=r"$\nabla_{\rm stable}$")
        ax.semilogy(r, res["nstar"], label=r"$\nabla^{*}$")
        ax.semilogy(r, res["nad"], ls="--", label=r"$\nabla_{\rm ad}$")
        ax.set_ylim(1e-1, 1e2)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(r"$r / R_0$")
        ax.set_ylabel(r"$\nabla$")
        ax.set_title("Temperature Gradients", loc="left")
        ax.legend(ncol=3, loc="upper right")
        plt.tight_layout()
        if savepath: fig.savefig(savepath)
        return fig

    def draw_cross_section(self, ax, title="Cross Section"):
        res, r = self._trim_result()
        L, L0 = res["L"], res["L"][0]
        
        # Color matching logic based on core properties
        colours = {(True, True): "#1f3b8a", (True, False): "#7fd0e3", (False, True): "#d62728", (False, False): "#ffd23f"}
        order = np.argsort(-r)
        
        prev_col = None
        for i in order:
            col = colours[(L[i] < 0.995 * L0, bool(res["is_conv"][i]))]
            if col != prev_col:
                ax.add_patch(Circle((0, 0), r[i], fc=col, ec=col, lw=0, zorder=int(10-r[i]*5)))
                prev_col = col

        lim = float(np.max(r)) * 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$r / R_0$")
        ax.set_ylabel(r"$r / R_0$")
        ax.set_title(title, loc="left")

    def plot_cross_section(self, savepath=None):
        fig, ax = plt.subplots(figsize=(7, 7))
        self.draw_cross_section(ax)
        ax.legend(handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=12, label=l)
            for c, l in [("#d62728", "Conv. Envelope"), ("#ffd23f", "Rad. Envelope"), ("#7fd0e3", "Core Zone")]
        ], loc="upper right")
        plt.tight_layout()
        if savepath: fig.savefig(savepath)
        return fig

    def plot_all(self, prefix="stellar_model"):
        self.plot_profiles(savepath=f"{prefix}_profiles.png")
        self.plot_gradients(savepath=f"{prefix}_gradients.png")
        self.plot_cross_section(savepath=f"{prefix}_cross.png")
        plt.show()
