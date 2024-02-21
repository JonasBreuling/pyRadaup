import numpy as np
import matplotlib.pyplot as plt
from pyRadaup import integration


def Robertson(DAE=False):
    """Robertson problem of semi-stable chemical reaction, see mathworks and Shampine2005.

    References:
    -----------
    mathworks: https://de.mathworks.com/help/matlab/math/solve-differential-algebraic-equations-daes.html#bu75a7z-5 \\
    Shampine2005: https://doi.org/10.1016/j.amc.2004.12.011
    """

    n = 3

    if DAE:
        var_index = np.array([0, 0, 1])
    else:
        var_index = np.zeros(n)
        # var_index = None

    if DAE:
        M = np.diag([1, 1, 0])
    else:
        M = np.eye(n)
        # M = None

    def f(t, y):
        y1, y2, y3 = y
        f = np.zeros(n, dtype=y.dtype)
        f[0] = -0.04 * y1 + 1e4 * y2 * y3
        f[1] = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
        if DAE:
            f[2] = y1 + y2 + y3 - 1
        else:
            f[2] = 3e7 * y2**2
        return f

    return M, f, var_index, n


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 1e-1
    t_span = (t0, t1)

    M, f, var_index, n = Robertson()

    # initial conditions
    y0 = np.array([1, 0, 0], dtype=float)
    y_dot0 = f(t0, y0)

    # solve the system
    bPrint = False
    rtol = 1e-4
    atol = 1e-12
    # sol = integration.radau5(
    # sol = integration.radaup(
    sol = integration.radau(
        tini=t0,
        tend=t1,
        y0=y0,
        fun=f,
        mljac=n,
        mujac=n,
        mlmas=0,
        mumas=0,
        rtol=rtol,
        atol=atol,
        t_eval=None,
        nmax_step=100000,
        max_step=t1,
        first_step=min(t1, 1e-6),
        max_ite_newton=7,
        # bUseExtrapolatedGuess=True,
        bUseExtrapolatedGuess=False,
        # bUsePredictiveController=True,
        bUsePredictiveController=False,
        safetyFactor=None,
        deadzone=None,
        step_evo_factor_bounds=None,
        jacobianRecomputeFactor=None,
        newton_tol=None,
        mass_matrix=M,
        var_index=var_index,
        bPrint=bPrint,
        nMaxBadIte=3,
        bAlwaysApply2ndEstimate=True,
        bReport=True,
    )

    t, y = sol.t, sol.y
    h = sol.h

    # visualization
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(t, y[0], "-b", label="y1")
    ax[0].plot(t, y[1] * 1e4, "-r", label="y2")
    ax[0].plot(t, y[2], "-y", label="y3")
    ax[0].set_xscale("log")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, h, "-ok", label="h")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].legend()
    ax[1].grid()

    plt.show()
