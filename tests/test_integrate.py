import numpy as np
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

    def M():
        if DAE:
            return np.diag([1, 1, 0])
        else:
            return None

    def fcn(n, x, y, f, rpar, ipar):
        y1, y2, y3 = y
        f[0] = -0.04 * y1 + 1e4 * y2 * y3
        f[1] = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
        if DAE:
            f[2] = y1 + y2 + y3 - 1
        else:
            f[2] = 3e7 * y2**2

    def jac(t, y):
        raise NotImplementedError

    return M, fcn, jac, var_index, n


if __name__ == "__main__":
    M, f, jac, var_index, n = Robertson()

    t0 = 0
    t1 = 1e-1
    h0 = 1e-3
    y0 = np.array([1, 0, 0], dtype=float)
    # y_dot0 = f(t0, y0)

    rtol = 1e-3
    atol = 1e-3
    itol = 0

    ijac = 0
    mjac = n
    mujac = n  # TODO:???

    imas = 1
    mlmas = n
    mumas = n  # TODO:???

    sol_t = []
    sol_y = []

    def solout(t, y):
        sol_t.append(y)
        sol_y.append(y)

    iout = 1

    # lwork = N*(LJAC+LMAS+NSMAX*LE+3*NSMAX+3)+20
    lwork = 100
    work = np.zeros(lwork, dtype=float)
    # liwork = (2+(NSMAX-1)/2)*N+20
    liwork = 100
    iwork = np.zeros(liwork, dtype=int)

    # rpar = np.zeros(1, dtype=float)
    # ipar = np.zeros(1, dtype=int)
    rpar = 0
    ipar = 0

    sol = radau.radau(
        n,
        f,
        t0,
        y0,
        t1,
        h0,
        rtol,
        atol,
        itol,
        jac,
        ijac,
        mjac,
        mujac,
        M,
        imas,
        mlmas,
        mumas,
        solout,
        iout,
        work,
        lwork,
        iwork,
        liwork,
        rpar,
        ipar,
    )
