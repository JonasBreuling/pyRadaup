#ifndef INTEGRATION_RADAU_H
#define INTEGRATION_RADAU_H

typedef void(*func_radaup)(int*, double*, double*, double*, double*, int*);

typedef void(*func_mas_radaup)(int*, double*, int*, int*, int*);

typedef void(*func_jac_radaup)(int*, double*, double*, double*, int*, double*, double*);

typedef void(*func_solout_radaup)(int*, double*, double*, double*, double*, int*, int*, double*, int*, int*);

void radaup_integration(double tini, double tend, double first_step,
                        int n, // size of the system
                        double *y0, // pointer to the initial solution vector
                        double *y, //
                        func_radaup fcn, // interface to the Python time derivative function
                        func_mas_radaup mas_fcn, // mass matrix evaluation function
                        func_solout_radaup solout, // solution export function
                        double rtol, double atol, // error tolerances (scalar only)
                        int mljac, int mujac, // Jacobian lower and upper bandwiths
                        int imas, int mlmas, int mumas, // Mass matrix lower and upper bandwiths
                        int *iwork_in, // integer parameters
                        double *work_in, // decimal parameters
                        int iout,  // solution export modevar_index
                        int *info // statistics
                        );

void radaup(int *n, func_radaup fcn, double *x, double *y, double *xend, double *h,
            double *rtol, double *atol, int *itol,
            void jac_radau(int*, double*, double*, double*, int*, double*, double*),
            int *ijac, int *mljac, int *mujac,
            func_mas_radaup mas_radau,
            int *imas, int *mlmas, int *mumas,
            func_solout_radaup solout,
            int *iout,
            double *work, int *lwork,int *iwork, int *liwork,
            double *rpar, int *ipar, int *idid);

void jac_radaup(int *n, double *x, double *y, double *dfy, int *ldfy, double *rpar, double *ipar);

#endif
