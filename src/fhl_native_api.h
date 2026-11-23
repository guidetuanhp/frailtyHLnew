#ifndef FHL_NATIVE_API_H
#define FHL_NATIVE_API_H

#include "fhl_core.h"

/* Cox Frailty Models (Gamma, Normal, AR(1)) */
SEXP fhl_fit_cox_gamma(
    SEXP s_x, SEXP s_z_list, SEXP s_y, SEXP s_del, SEXP s_Mi,
    SEXP s_idx2, SEXP s_t2, SEXP s_di,
    SEXP s_beta_init, SEXP s_v_init, SEXP s_alpha_init,
    SEXP s_mord, SEXP s_dord, SEXP s_varfixed, SEXP s_varnonneg,
    SEXP s_max_iter, SEXP s_convergence
);

SEXP fhl_fit_cox_normal(
    SEXP s_x, SEXP s_z_list, SEXP s_y, SEXP s_del, SEXP s_Mi,
    SEXP s_idx2, SEXP s_t2, SEXP s_di,
    SEXP s_beta_init, SEXP s_v_init, SEXP s_alpha_init,
    SEXP s_mord, SEXP s_dord, SEXP s_varfixed, SEXP s_varnonneg,
    SEXP s_max_iter, SEXP s_convergence
);

SEXP fhl_fit_cox_ar1(
    SEXP x_s, SEXP z_list_s, SEXP y_s, SEXP del_s, SEXP Mi_s,
    SEXP idx2_s, SEXP t2_s, SEXP di_s,
    SEXP beta_init_s, SEXP v_init_s, SEXP alpha_init_s,
    SEXP mord_s, SEXP dord_s, SEXP varfixed_s, SEXP varnonneg_s,
    SEXP max_iter_s, SEXP convergence_s
);

/* Competing Risks Frailty Model */
SEXP fhl_fit_competing_risks(
    SEXP X_s, SEXP group_s, SEXP n_s, SEXP time_s, SEXP index_s,
    SEXP y_list, SEXP d_list, SEXP beta_init_s, SEXP theta_init_s,
    SEXP v_init_s, SEXP order_s, SEXP cov_type_s, SEXP subhazard_s,
    SEXP max_iter_s, SEXP tol_s
);

/* Mixed-Effects AFT Models */
SEXP fhl_fit_aft_1re(
    SEXP y_s, SEXP x_s, SEXP z_or_group_s, SEXP qq_s,
    SEXP delta_s, SEXP beta_h0_s, SEXP v_h0_s, SEXP maxiter_s
);

SEXP fhl_fit_aft_2re(
    SEXP y_s, SEXP x_s,
    SEXP z1_or_group1_s, SEXP qq1_s,
    SEXP z2_or_group2_s, SEXP qq2_s,
    SEXP delta_s, SEXP beta_h0_s, SEXP v_h0_s, SEXP maxiter_s
);

/* Joint Models */
SEXP fhl_fit_jm_cox1(
    SEXP y1_s, SEXP X1_s, SEXP Z1_s,
    SEXP del2_s, SEXP X2_s, SEXP Z2_s,
    SEXP di_s, SEXP Mi_s, SEXP idx2_s,
    SEXP beta_h0_s, SEXP v_h0_s, SEXP maxiter_s
);

SEXP fhl_fit_jm_cox2(
    SEXP y1_s, SEXP X1_s, SEXP Z1_s,
    SEXP del2_s, SEXP X2_s, SEXP Z2_s,
    SEXP di2_s, SEXP Mi2_s, SEXP idx2_2_s, SEXP beta_h0_s,
    SEXP del3_s, SEXP X3_s, SEXP Z3_s,
    SEXP di3_s, SEXP Mi3_s, SEXP idx2_3_s, SEXP beta_h02_s,
    SEXP v_h0_s, SEXP maxiter_s
);

SEXP fhl_fit_jm_frailty(
    SEXP del1_s, SEXP X1_s, SEXP Z1_s,
    SEXP di1_s, SEXP Mi1_s, SEXP idx2_1_s, SEXP beta_h01_s,
    SEXP del2_s, SEXP X2_s, SEXP Z2_s,
    SEXP di2_s, SEXP Mi2_s, SEXP idx2_2_s, SEXP beta_h02_s,
    SEXP v_h0_s, SEXP maxiter_s
);

/* Variable Selection in Cox Frailty Models */
SEXP fhl_fit_vs_1re(
    SEXP X_s, SEXP B_s, SEXP Z_list_s, SEXP v_list_s,
    SEXP pen_type_s, SEXP alpha_s,
    SEXP tunset1_s, SEXP tunset2_s,
    SEXP di_s, SEXP Mi_s, SEXP del_s, SEXP varfixed_s
);

SEXP fhl_fit_vs_2re(
    SEXP X_s, SEXP B_s, SEXP Z_list_s, SEXP v_list_s,
    SEXP pen_type_s, SEXP alpha_s,
    SEXP tunset1_s, SEXP tunset2_s,
    SEXP di_s, SEXP Mi_s, SEXP del_s
);

#endif /* FHL_NATIVE_API_H */
