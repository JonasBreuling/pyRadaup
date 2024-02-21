from pyRadaup import integration
import numpy as np
import matplotlib.pyplot as plt

# Test the pendulum
# from scipy.integrate import solve_ivp
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_equal,
    assert_no_warnings,
    suppress_warnings,
)
import matplotlib.pyplot as plt

"""
Created on Mon Nov  2 14:54:33 2020

1) Modelling the pendulum

  The pendulum with constant rod-length may be represented in various manner:
  DAEs of index 3 to 0. The present script implement each possibility and compares
  the corresponding solution to the true solution obtained from the simple
  pendulum ODE y'' = -g*sin(y).

  The pendulum DAE of index 3 is obtained simply by using Newton's law on the
  pendulum of mass m in the (x,y) reference frame, x being the horizontal axis,
  and y the vertical axis (positive upwards). The rod force exerced on the mass
  is T. The system reads:

    dx/dt = vx                         (1)
    dydt = vy                          (2)
    d(vx)/dt = -(T/m)*sin(theta)       (3)
    d(vy)/dt =  (T/m)*cos(theta) - g   (4)
    x^2 + y^2 = r0^2                   (5)

  Equation (5) ensures that the rod-length remains constant.
  By expression sin(theta) as x/sqrt(x**2+y**2) and cos(theta)=-y/sqrt(x**2+y**2),
  we may introduce lbda=T/sqrt(x**2+y**2) and rewrite Equations (4) and (5) as:

    d(vx)/dt = -lbda*x/m               (3a)
    d(vy)/dt = -lbda*y/m - g           (4a)

  The new variable "lbda" plays the role of a Lagrange multiplier, which adjusts
  istelf in time so that equation (5) is not violated, i.e. the solution remains
  on the manifold x^2 +y^2 - r0^2 = 0.

  The corresponding system (1,2,3a,4a,5) is a DAE of index 3.
  We can obtain lower-index DAE by differentiating equation (5) with respect
  to time. If we do it once, we obtain:

    x*vx + y*vy = 0                    (6)

  Physically, this means that the velocity vector of the pendulum mass is
  always perpendicular to the rod. The system (1,2,3a,3b,6) is a DAE of index 1.
  If we differentiate it once more with respect to time, we obtain:

    lbda*(x^2 + y^2)  + m*(vx^2 + vy^2) - m*g*y = 0     (7)

  The system(1,2,3a,4a,7) is a DAE of index 1. Note that we could directly
  remove lambda from the problem as we now have its expression. Physically,
  equation (7) means that the rod force must equilibrate the centrifugal force
  and the gravity force, so that the mass remains attache at the rod.
  We may differentiate (7) once more to obtain a (long) ODE on lbda:

    d(lbda)/dt = f(x,y,vx,vy)          (8)

  The system (1,2,3a,4a,8) is a DAE Of index 0, i.e. an ODE ! This is actually
  how we define the original index of the first system: the number of times
  equation (5) must be derived so that only ODEs remain.

  2) Numerical solution

    The adapted version of Scipy's Radau, RadauDAE, is able to solve problems
    of the form:
      M y' = f(t,y)       (9)
    even when the mass matrix M is singular. This enables the computation of
    the solution of DAEs expressed in Hessenberg form.

    Radau5 is originally able to adapt the time step thanks to a 3rd-order
    error estimate. This estimate is however not suited for DAEs, as the error
    on DAEs is vastly overestimated (see [1]). Hence in this implementation, the
    error on the algebraic variables is forced to zero, when these are clearly
    identified (those that correspond to a row of zeros in M). Note that [1]
    recommends to multiply the algebraic error by the time step instead.

    Solution of (9) is also possible when M is singular, but without any rows
    full of zeros (see the transistor amplifier example in this git, based on
    [2]).

    Additional information can be printed during the Radau computation by setting
    bPrint to True:
      newton convergence, norm of the error estimate...

  References:
    [1] Hairer, Ernst & Lubich, Christian & Roche, Michel. (1989)
        The numerical solution of differential-algebraic systems
        by Runge-Kutta methods, Springer
    [2] Hairer, Ernst & Wanner, Gerhard. (1996)
        Solving Ordinary Differential Equations II:
        Stiff and Differential-Algebraic Problems

@author: laurent
"""
import numpy as np


def computeAngle(x, y):
    theta = np.arctan(-x / y)
    I = np.where(y > 0)[0]
    theta[I] += np.sign(x[I]) * np.pi
    return theta


#%% Setup the model based on the chosen formulation
def generateSystem(
    chosen_index, theta_0=np.pi / 2, theta_dot0=0.0, r0=1.0, m=1, g=9.81
):
    """Generates the DAE function representing the pendulum with the desired
    index (0 to 3).
        Inputs:
          chosen_index: int
            index of the DAE formulation
          theta_0: float
            initial angle of the pendulum in radians
          theta_dot0: float
            initial angular velocity of the pendulum (rad/s)
          r0: float
            length of the rod
          m: float
            mass of the pendulum
          g: float
            gravitational acceleration (m/s^2)
        Outputs:
          dae_fun: callable
            function of (t,x) that represents the system. Will be given to the
            solver.
          jac_dae: callable
            jacobian of dae_fun with respect to x. May be given to the solver.
          mass: array_like
            mass matrix
          Xini: initial condition for the system

    """

    if chosen_index == "GGL":

        def dae_fun(t, X):
            # X= (x,y,xdot=vx, ydot=vy, lbda)
            x = X[0]
            y = X[1]
            vx = X[2]
            vy = X[3]
            lbda = X[4]
            mu = X[5]
            return np.array(
                [
                    vx - x * mu,
                    vy - y * mu,
                    -x * lbda / m,
                    -g - (y * lbda) / m,
                    x * vx + y * vy,
                    x**2 + y**2 - r0**2,
                ]
            )

        mass = np.eye(6)  # mass matrix M
        mass[-2, -2] = 0
        mass[-1, -1] = 0
        var_index = np.array([0, 0, 0, 0, 2, 3])
        jac_dae = None
    elif chosen_index == 3:

        def dae_fun(t, X):
            # X= (x,y,xdot=vx, ydot=vy, lbda)
            x = X[0]
            y = X[1]
            vx = X[2]
            vy = X[3]
            lbda = X[4]
            return np.array(
                [vx, vy, -x * lbda / m, -g - (y * lbda) / m, x**2 + y**2 - r0**2]
            )

        mass = np.eye(5)  # mass matrix M
        mass[-1, -1] = 0
        var_index = np.array([0, 0, 0, 0, 3])

        def jac_dae(t, X):
            x = X[0]
            y = X[1]
            vx = X[2]
            vy = X[3]
            lbda = X[4]
            return np.array(
                [
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [-lbda / m, 0.0, 0.0, 0.0, -x / m],
                    [0.0, -lbda / m, 0.0, 0.0, -y / m],
                    [2 * x, 2 * y, 0.0, 0.0, 0.0],
                ]
            )

    elif chosen_index == 2:

        def dae_fun(t, X):
            x = X[0]
            y = X[1]
            vx = X[2]
            vy = X[3]
            lbda = X[4]
            return np.array(
                [
                    vx,
                    vy,
                    -x * lbda / m,
                    -g - (y * lbda) / m,
                    x * vx + y * vy,
                ]
            )

        mass = np.eye(5)
        mass[-1, -1] = 0
        var_index = np.array([0, 0, 0, 0, 2])

        def jac_dae(t, X):
            x = X[0]
            y = X[1]
            vx = X[2]
            vy = X[3]
            lbda = X[4]
            return np.array(
                [
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [-lbda / m, 0.0, 0.0, 0.0, -x / m],
                    [0.0, -lbda / m, 0.0, 0.0, -y / m],
                    [vx, vy, x, y, 0.0],
                ]
            )

    elif chosen_index == 1:

        def dae_fun(t, X):
            x = X[0]
            y = X[1]
            vx = X[2]
            vy = X[3]
            lbda = X[4]
            return np.array(
                [
                    vx,
                    vy,
                    -x * lbda / m,
                    -g - (y * lbda) / m,
                    lbda * (x**2 + y**2) / m + g * y - (vx**2 + vy**2),
                ]
            )

        mass = np.eye(5)
        mass[-1, -1] = 0
        var_index = np.array([0, 0, 0, 0, 1])

        def jac_dae(t, X):
            x = X[0]
            y = X[1]
            vx = X[2]
            vy = X[3]
            lbda = X[4]
            return np.array(
                [
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [-lbda / m, 0.0, 0.0, 0.0, -x / m],
                    [0.0, -lbda / m, 0.0, 0.0, -y / m],
                    [
                        2 * x * lbda / m,
                        2 * y * lbda / m + g,
                        -2 * vx,
                        -2 * vy,
                        (x**2 + y**2) / m,
                    ],
                ]
            )

    elif chosen_index == 0:

        def dae_fun(t, X):
            x = X[0]
            y = X[1]
            vx = X[2]
            vy = X[3]
            lbda = X[4]
            dvx = -lbda * x / m
            dvy = -lbda * y / m - g
            rsq = x**2 + y**2  # = r(t)^2
            dt_lbda = (1 / m) * (
                -g * vy / rsq
                + 2 * (vx * dvx + vy * dvy) / rsq
                + (vx**2 + vy**2 - g * y) * (2 * x * vx + 2 * y * vy) / (rsq**2)
            )
            return np.array([vx, vy, dvx, dvy, dt_lbda])

        mass = None  # or np.eye(5) the identity matrix
        var_index = np.array([0, 0, 0, 0, 0])
        jac_dae = None  # use the finite-difference routine, this expression would
        # otherwise be quite heavy :)
    else:
        raise Exception("index must be in [0,3]")

    # alternatively, define the Jacobian via finite-differences, via complex-step
    # to ensure machine accuracy
    if jac_dae is None:
        import scipy.optimize._numdiff

        jac_dae = lambda t, x: scipy.optimize._numdiff.approx_derivative(
            fun=lambda x: dae_fun(t, x),
            x0=x,
            method="cs",
            rel_step=1e-50,
            f0=None,
            bounds=(-np.inf, np.inf),
            sparsity=None,
            as_linear_operator=False,
            args=(),
            kwargs={},
        )
    ## Otherwise, the Radau solver uses its own routine to estimate the Jacobian, however the
    # original Scip routine is only adapted to ODEs and may fail at correctly
    # determining the Jacobian because of it would chosse finite-difference step
    # sizes too small with respect to the problem variables (<1e-16 in relative).
    # jac_dae = None

    ## Initial condition (pendulum at an angle)
    x0 = r0 * np.sin(theta_0)
    y0 = -r0 * np.cos(theta_0)
    vx0 = r0 * theta_dot0 * np.cos(theta_0)
    vy0 = r0 * theta_dot0 * np.sin(theta_0)
    lbda_0 = (
        m * r0 * theta_dot0**2 + m * g * np.cos(theta_0)
    ) / r0  # equilibrium along the rod's axis
    mu_0 = 0

    # the initial condition should be consistent with the algebraic equations
    if chosen_index == "GGL":
        Xini = np.array([x0, y0, vx0, vy0, lbda_0, mu_0])
    else:
        Xini = np.array([x0, y0, vx0, vy0, lbda_0])

    return dae_fun, jac_dae, mass, Xini, var_index


###### Parameters to play with
chosen_index = 3  # The index of the DAE formulation
tf = 5.0  # 10.0        # final time (one oscillation is ~2s long)
rtol = 1e-5
atol = rtol  # relative and absolute tolerances for time adaptation
first_step = 1e-2
# dt_max = np.inf
# rtol=1e-3; atol=rtol # relative and absolute tolerances for time adaptation
bPrint = False  # if True, additional printouts from Radau during the computation
bDebug = False  # study condition number of the iteration matrix


## Physical parameters for the pendulum
theta_0 = np.pi / 2  # initial angle
theta_dot0 = -1.0  # initial angular velocity
r0 = 3.0  # rod length
m = 1.0  # mass
g = 9.81  # gravitational acceleration

dae_fun, jac_dae, mass, Xini, var_index = generateSystem(
    chosen_index, theta_0, theta_dot0, r0, m, g
)
n = Xini.size
# var_index = 3+ 0*var_index
# jac_dae = None
print(f"Solving the index {chosen_index} formulation")

# for isolv, name in ((2, "scipyDAE"),):  # (1, 'pyRadau5'),
for isolv, name in ((1, "pyRadau5"),):
    if isolv == 1:
        #%% Solve the DAE
        solfort = integration.radau5(
            tini=0.0,
            tend=tf,
            y0=Xini,
            fun=dae_fun,
            mljac=n,
            mujac=n,
            mlmas=0,
            mumas=0,
            rtol=rtol,
            atol=atol,
            t_eval=None,
            nmax_step=100000,
            max_step=tf,
            first_step=min(tf, first_step),
            max_ite_newton=7,
            bUseExtrapolatedGuess=True,
            bUsePredictiveController=True,
            safetyFactor=None,
            deadzone=None,
            step_evo_factor_bounds=None,
            jacobianRecomputeFactor=None,
            newton_tol=None,
            mass_matrix=mass,
            var_index=var_index,
            bPrint=bPrint,
            nMaxBadIte=3,
            bAlwaysApply2ndEstimate=True,
            bReport=True,
        )
        sol = solfort
        if solfort.success:
            state = "solved"
        else:
            state = "failed"
        print("Fortran DAE of index {} {}".format(chosen_index, state))
        print(
            "{} time steps ({} = {} accepted + {} rejected + {} failed)".format(
                sol.t.size - 1, sol.nstep, sol.naccpt, sol.nrejct, sol.nfail
            )
        )
        print(
            "{} fev, {} jev, {} LUdec, {} linsolves, {} linsolves for error estimation".format(
                sol.nfev, sol.njev, sol.ndec, sol.nsol, sol.nlinsolves_err
            )
        )

        # recover the time history of each variable
        x, y, vx, vy, lbda = sol.y
        T = lbda * np.sqrt(x**2 + y**2)
        theta = computeAngle(x, y)
        dicfort = {
            "t": sol.t,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "theta": theta,
            "lbda": lbda,
            "T": T,
        }
    elif isolv == 2:
        #%% Solve the DAE with Scipy's modified Radau
        var_index = np.array([0, 0, 0, 0, 3])
        from scipyDAE.radauDAE import RadauDAE
        from scipyDAE.radauDAE import solve_ivp_custom as solve_ivp

        solpy = solve_ivp(
            fun=dae_fun,
            t_span=(0.0, tf),
            y0=Xini,
            max_step=tf,
            rtol=rtol,
            atol=atol,
            jac=jac_dae,
            jac_sparsity=None,
            method=RadauDAE,
            first_step=min(tf, first_step),
            dense_output=True,
            mass_matrix=mass,
            bPrint=bPrint,
            max_newton_ite=8,
            min_factor=0.2,
            max_factor=10,
            factor_on_non_convergence=0.5,
            var_index=var_index,
            # newton_tol=1e-4,
            scale_residuals=True,
            scale_newton_norm=True,
            scale_error=True,
            zero_algebraic_error=False,
            bAlwaysApply2ndEstimate=True,
            max_bad_ite=1,
            bUsePredictiveNewtonStoppingCriterion=False,
            bDebug=bDebug,
            bReport=True,
            bPrintProgress=True,
        )
        sol = solpy
        sol.reports = sol.solver.reports
        for key in sol.reports.keys():
            sol.reports[key] = np.array(sol.reports[key])
        if sol.success:
            state = "solved"
        else:
            state = "failed"
        print("\nScipy DAE of index {} {}".format(chosen_index, state))
        print(
            "{} time steps ({} = {} accepted + {} rejected + {} failed)".format(
                sol.t.size - 1,
                sol.solver.nstep,
                sol.solver.naccpt,
                sol.solver.nrejct,
                sol.solver.nfailed,
            )
        )
        print(
            "{} fev, {} jev, {} LUdec, {} linsolves, {} linsolves for error estimation".format(
                sol.nfev,
                sol.njev,
                sol.nlu,
                sol.solver.nlusolve,
                sol.solver.nlusolve_errorest,
            )
        )

        x, y, vx, vy, lbda = sol.y
        T = lbda * np.sqrt(x**2 + y**2)
        theta = computeAngle(x, y)
        dicpy = {
            "t": sol.t,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "theta": theta,
            "lbda": lbda,
            "T": T,
        }

    #%% Plot report
    print("codes = ", np.unique(sol.reports["code"]))

    codes, names = zip(
        *(
            (0, "accepted"),
            (1, "rejected"),
            (2, "refactor_from_newton"),
            (3, "update_from_newton"),
            (4, "singular"),
            (5, "badconvergence"),
            (-10, "jacobian_update"),
            (-11, "refactor"),
        )
    )
    Idict = {}
    for key, val in zip(names, codes):
        Idict[key] = np.where(sol.reports["code"] == val)[0]
    Idict["failed"] = np.where(
        np.logical_or(sol.reports["code"] == 4, sol.reports["code"] == 5)
    )[0]

    plt.figure()
    plt.plot(sol.t[:-1], np.diff(sol.t), label=r"$\Delta t$")
    # plt.plot(sol.reports['t'][Idict['accepted']], sol.reports['dt'][Idict['accepted']], linestyle='--', label='dt2')

    # plt.plot(sol.reports['t'][Idict['refactor']],        sol.reports['dt'][Idict['refactor']], linestyle='', marker='o', color='tab:green', label='refactor', markersize=15, alpha=0.3)
    # plt.plot(sol.reports['t'][Idict['jacobian_update']], sol.reports['dt'][Idict['jacobian_update']], linestyle='', marker='o', color='tab:green', label='jacobian update', alpha=1)

    plt.plot(
        sol.reports["t"][Idict["jacobian_update"]],
        sol.reports["dt"][Idict["jacobian_update"]],
        linestyle="",
        marker="o",
        color="tab:green",
        label="jacobian update",
        markersize=15,
        alpha=0.3,
    )
    plt.plot(
        sol.reports["t"][Idict["failed"]],
        sol.reports["dt"][Idict["failed"]],
        linestyle="",
        marker="o",
        color="tab:red",
        label="failed",
    )
    plt.plot(
        sol.reports["t"][Idict["rejected"]],
        sol.reports["dt"][Idict["rejected"]],
        linestyle="",
        marker="o",
        color="tab:purple",
        label="rejected",
    )
    plt.legend()
    plt.yscale("log")

    ax2 = plt.gca().twinx()
    ax2.plot(
        sol.reports["t"][Idict["accepted"]],
        sol.reports["newton_iterations"][Idict["accepted"]],
        label="all",
        color="tab:green",
        linestyle="--",
    )
    ax2.plot(
        sol.reports["t"][Idict["accepted"]],
        sol.reports["bad_iterations"][Idict["accepted"]],
        label="bad",
        color="tab:red",
        linestyle="--",
    )
    ax2.legend()
    ax2.set_ylabel("Newton iterations")
    plt.grid()
    plt.legend()
    plt.xlabel("t (s)")
    plt.ylabel("dt (s)")
    plt.title(f"{name} analysis")

    #%% Time step evolution
    plt.figure()
    plt.semilogy(
        sol.reports["t"][Idict["accepted"]],
        sol.reports["err1"][Idict["accepted"]],
        label="err1",
        color="tab:green",
        linestyle="-",
    )
    plt.semilogy(
        sol.reports["t"][Idict["accepted"]],
        sol.reports["err2"][Idict["accepted"]],
        label="err2",
        color="tab:orange",
        linestyle="-",
    )

    plt.semilogy(
        sol.reports["t"][Idict["rejected"]],
        sol.reports["err1"][Idict["rejected"]],
        label="rejected err1",
        color="tab:green",
        linestyle="",
        marker=".",
    )
    plt.semilogy(
        sol.reports["t"][Idict["rejected"]],
        sol.reports["err2"][Idict["rejected"]],
        label="rejected err2",
        color="tab:orange",
        linestyle="",
        marker=".",
    )
    plt.axhline(1.0, linestyle="--", label=None, color=[0, 0, 0])
    plt.grid()
    plt.legend()
    plt.xlabel("t (s)")
    plt.ylabel("error")
    plt.title(f"{name} error estimates")

    #%% Component-wise errors
    if "errors1" in sol.reports.keys():
        plt.figure()
        for ivar in range(sol.y.shape[0]):
            varname = ["x", "y", "vx", "vy", "lbda"][ivar]
            (p,) = plt.semilogy(
                sol.reports["t"][Idict["accepted"]],
                np.abs(sol.reports["errors1"][Idict["accepted"], ivar]),
                label=varname,
                linestyle="-",
            )
            plt.semilogy(
                sol.reports["t"][Idict["accepted"]],
                np.abs(sol.reports["errors2"][Idict["accepted"], ivar]),
                label=None,
                color=p.get_color(),
                linestyle="--",
            )
        plt.axhline(1.0, linestyle="--", label=None, color=[0, 0, 0])
        plt.plot(np.nan, linestyle="-", label="err1", color=[0, 0, 0])
        plt.plot(np.nan, linestyle="--", label="err2", color=[0, 0, 0])
        plt.grid()
        plt.legend(ncol=5, framealpha=0.3, loc="lower center")
        plt.xlabel("t (s)")
        plt.ylabel("error")
        plt.title(f"{name} error estimates")
        plt.ylim(1e-12, None)
    #%% Ratio of both error estimates
    if "errors1" in sol.reports.keys():
        plt.figure()
        for ivar in range(sol.y.shape[0]):
            varname = ["x", "y", "vx", "vy", "lbda"][ivar]
            (p,) = plt.semilogy(
                sol.reports["t"][Idict["accepted"]],
                np.abs(sol.reports["errors2"][Idict["accepted"], ivar])
                / np.abs(sol.reports["errors1"][Idict["accepted"], ivar]),
                label=varname,
                linestyle="-",
            )
        plt.axhline(1.0, linestyle="--", label=None, color=[0, 0, 0])
        plt.grid()
        plt.legend(ncol=5, framealpha=0.3, loc="lower center")
        plt.xlabel("t (s)")
        plt.ylabel("error")
        plt.title(f"{name} ratio of both error estimates")
        # plt.ylim(1e-12, None)

    #%% Ratio of both error estimates
    if "error_scale" in sol.reports.keys():
        plt.figure()
        for ivar in range(sol.y.shape[0]):
            varname = ["x", "y", "vx", "vy", "lbda"][ivar]
            (p,) = plt.semilogy(
                sol.reports["t"][Idict["accepted"]],
                np.abs(sol.reports["error_scale"][Idict["accepted"], ivar]),
                label=varname,
                linestyle="-",
            )
        plt.axhline(1.0, linestyle="--", label=None, color=[0, 0, 0])
        plt.grid()
        plt.legend(ncol=5, framealpha=0.3, loc="lower center")
        plt.xlabel("t (s)")
        plt.ylabel("error")
        plt.title(f"{name} error scales")
        # plt.ylim(1e-12, None)

#%%
raise Exception("stop")
#%% Compute true solution (ODE on the angle in polar coordinates)
def fun_ode(t, X):
    theta = X[0]
    theta_dot = X[1]
    return np.array([theta_dot, -g / r0 * np.sin(theta)])


Xini_ode = np.array([theta_0, theta_dot0])
sol_ode = solve_ivp(
    fun=fun_ode,
    t_span=(0.0, tf),
    y0=Xini_ode,
    rtol=1e-12,
    atol=1e-12,
    max_step=tf / 10,
    method="DOP853",
    dense_output=True,
)
theta_ode = sol_ode.y[0, :]
theta_dot = sol_ode.y[1, :]
x_ode = r0 * np.sin(theta_ode)
y_ode = -r0 * np.cos(theta_ode)
vx_ode = r0 * theta_dot * np.cos(theta_ode)
vy_ode = r0 * theta_dot * np.sin(theta_ode)
T_ode = m * r0 * theta_dot**2 + m * g * np.cos(theta_ode)
lbda_ode = T_ode / np.sqrt(x_ode**2 + y_ode**2)

dicref = {
    "t": sol_ode.t,
    "x": x_ode,
    "y": y_ode,
    "vx": vx_ode,
    "vy": vy_ode,
    "theta": theta_ode,
    "lbda": lbda_ode,
    "T": T_ode,
}


#%% Compare the DAE solution and the true solution
plt.figure()
for sol, name, linestyle, marker, color in [
    # (dicfort,'fortran','-','+','tab:blue'),
    (dicpy, "py", ":", "o", "tab:orange"),
    (dicref, "ref", "--", None, "tab:green"),
]:
    plt.plot(
        sol["t"], sol["x"], color=color, linestyle=linestyle, marker=marker, label=None
    )  # r'$x_{{}}$'.format(name))
    # plt.plot(sol.t,y, color='tab:blue', linestyle='--', label=r'$y_{DAE}$')
    plt.plot(
        sol["t"],
        sol["y"] ** 2 + sol["x"] ** 2,
        color=color,
        marker=marker,
        linestyle=linestyle,
        label=None,
    )  # r'$x_{{}}^2 + y_{{}}^2$'.format(name,name))
    plt.plot(
        np.nan, np.nan, linestyle=linestyle, color=color, marker=marker, label=name
    )
plt.grid()
plt.legend()
plt.title("Comparison with the true solution")

#%% Analyze constraint violations
# we check how well equations (5), (6) and (7) are respected
fig, ax = plt.subplots(4, 1, sharex=True)
for sol, name, linestyle, marker, color in [
    # (dicfort,'fortran','-','+','tab:blue'),
    (dicpy, "py", ":", ".", "tab:orange"),
    # (dicref, 'ref', '--',None,'tab:green')
]:
    for i in range(3):
        t, x, y, vx, vy, lbda = (
            sol["t"],
            sol["x"],
            sol["y"],
            sol["vx"],
            sol["vy"],
            sol["lbda"],
        )
        if i == 0:
            constraint = (
                lbda * (x**2 + y**2) / m + g * y - (vx**2 + vy**2)
            )  # index 1
        if i == 1:
            constraint = x * vx + y * vy  # index 2
        if i == 2:
            constraint = x**2 + y**2 - r0**2  # index 3

        # ax[i].plot(t, constraint, marker=marker, linestyle=linestyle, label=name, color=color)
        ax[i].semilogy(
            t,
            np.abs(constraint),
            marker=marker,
            linestyle=linestyle,
            label=name,
            color=color,
        )

    ax[3].semilogy(
        t[:-1], np.diff(t), marker=marker, linestyle=linestyle, label=name, color=color
    )
for i in range(4):
    ax[i].grid()
for i in range(3):
    ax[i].set_ylabel("index {}".format(i + 1))
ax[3].set_ylabel("dt")
ax[-1].legend()
ax[-1].set_xlabel("t (s)")
fig.suptitle("Constraints violation")
