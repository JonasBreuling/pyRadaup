import numpy as np
import matplotlib.pyplot as plt
from cardillo.math.approx_fprime import approx_fprime
import radau


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

    def mas(am, rpar, ipar, n, lmas):
        print(f"mas called")
        # am = np.eye(3)
        am[0, 0] = 1
        am[1, 1] = 1
        am[2, 2] = 1
        if not DAE:
            am[2, 2] = 1
        print(f"mas finished")

    def fcn(x, y, f, rpar, ipar, n):
        # print(f"fcn called")
        y1, y2, y3 = y
        # c = 1e-3
        # f[:] = -c * y[:]
        # f = np.zeros(n, dtype=float)
        f[0] = -0.04 * y1 + 1e4 * y2 * y3
        f[1] = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
        # f[2] = 3e7 * y2**2
        if DAE:
            f[2] = y1 + y2 + y3 - 1
        else:
            f[2] = 3e7 * y2**2
        # print(f"fcn finished")

    def jac(t, y, fjac, rpar, ipar, n, ldjac):
        # raise NotImplementedError
        # print(f"jac called")
        def fun(y):
            f = np.zeros(3)
            fcn(t, y, f, rpar, ipar, n)
            return f

        fjac = approx_fprime(y, fun, method="2-point")
        # print(f"jac finished")

    return mas, fcn, jac, var_index, n


if __name__ == "__main__":
    mass, fcn, jac, var_index, n = Robertson()

    t0 = 0
    t1 = 1e8
    h0 = 1e-8
    y0 = np.array([1.0, 0, 0], dtype=float)

    rtol = 1e-14
    atol = 1e-14
    itol = 0

    ijac = 0  # numerical jacobian
    # ijac = 1 # user-defined jacobian
    mljac = n
    mujac = n  # TODO:???

    imas = 0  # assume identity mas
    # imas = 1 # user-defined mas
    mlmas = n
    mumas = n  # TODO:???

    sol_t = []
    sol_h = []
    sol_y = []

    def solout(nrsol, xosol, xsol, y, cont, rpar, ipar, irtrn, lrc, nsolu):
        # print(f"solout called")
        # cont(0, (xsol - xosol) / 2, )
        # TODO: Compute continuous output using the collocation polynomial, similar to fortran CONTRA function.
        print(f"t: {xsol}; y: {y}")
        sol_t.append(xsol)
        sol_h.append(xsol - xosol)
        sol_y.append(y.copy())

    iout = 1

    # setup work array
    ljac = n
    lmas = n
    # TODO: Why only (5,5) and (7,7) are working but default (3,7) not?
    # nsmin = 5
    # nsmax = 5
    # nsmin = 7
    # nsmax = 7
    nsmin = 5
    nsmax = 7
    le = n
    lwork = n * (ljac + lmas + nsmax * le + 3 * nsmax + 3) + 20
    # lwork += 100
    work = np.zeros(lwork, dtype=float)
    # work[6] = 1e-0 # maximum step size

    liwork = int(2 + (nsmax - 1) / 2) * n + 20
    # liwork += 100
    iwork = np.zeros(liwork, dtype=int)
    iwork[1] = int(1e6)  # maximum number of allowed steps
    iwork[10] = nsmin
    iwork[11] = nsmax
    # iwork[12] = nsmin

    # rpar = np.zeros(10, dtype=float)
    # ipar = np.zeros(10, dtype=int)
    rpar = 0
    ipar = 0

    idid = 1

    print(radau.radau.__doc__)

    sol = radau.radau(
        fcn,
        t0,
        y0,
        t1,
        h0,
        rtol,
        atol,
        itol,
        jac,
        ijac,
        mljac,
        mujac,
        mass,
        imas,
        mlmas,
        mumas,
        solout,
        iout,
        work,
        iwork,
        rpar,
        ipar,
        idid,
        # optional arguments
        # n,
        # lwork,
        # liwork,
    )

    # visualization
    t = np.array(sol_t)
    h = np.array(sol_h)
    y = np.array(sol_y).T
    fig, ax = plt.subplots(2, 1)

    print(f"t: {t}")
    print(f"y:\n{y}")

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
