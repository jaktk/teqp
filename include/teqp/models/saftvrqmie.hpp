#pragma once

/**
 SAFT-VRQ Mie: the Feynman-Hibbs corrected SAFT-VR Mie equation of state for spherical molecules of
 
 A. Aasen, M. Hammer, Å. Ervik, E. A. Müller, Ø. Wilhelmsen, "Equation of state and force fields for
 Feynman-Hibbs-corrected Mie fluids. I. Application to pure helium, neon, hydrogen, and deuterium",
 J. Chem. Phys. 151, 064508 (2019), https://doi.org/10.1063/1.5111364
 
 A. Aasen, M. Hammer, E. A. Müller, Ø. Wilhelmsen, "... II. Application to mixtures of helium, neon,
 hydrogen, and deuterium", J. Chem. Phys. 152, 074507 (2020), https://doi.org/10.1063/1.5136079
 
 Where the printed equations and the reference implementation in thermopack (SINTEF, MIT/Apache-2.0,
 https://github.com/thermotools/thermopack, commit 1884204, version 2.2.3) differ, this follows
 thermopack: the prefactor x0 = sigma_ij/d_ij with I and J at sigma_eff,ij/d_ij, the factor 2 on the
 cross terms of u^2, the Carnahan-Starling residual for the pure hard-sphere fluid, and the signs of
 a2 and a3. Barker-Henderson diameters are integrated for every pair from the cross potential with
 the 21-point Gauss-Kronrod rule, and the hard-sphere term is the non-additive mixture of Santos et al.
 */

#include "teqp/types.hpp"
#include "teqp/constants.hpp"
#include "teqp/exceptions.hpp"
#include "nlohmann/json.hpp"
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace teqp::SAFTVRQMie {

/// Parameters of one spherical (m = 1) fluid
struct PureParameters{
    std::string name;
    double sigma_m, epsilon_over_k, lambda_a, lambda_r, mass_kg;
};

/// Parameters of one pair (i, j), i == j included; combining rules of Part II Eqs. 2-8
struct PairParameters{
    double sigma;     ///< sigma_ij in m, with l_ij
    double eps;       ///< epsilon_ij/k_B in K, with k_ij
    double la, lr;    ///< lambda_a,ij and lambda_r,ij
    double C;         ///< Mie prefactor
    double Q1a, Q1r, Q2a, Q2r;  ///< Feynman-Hibbs coefficients Q1(lambda) = lambda(lambda-1), Q2 = 1/2 lambda(lambda-1)(lambda+1)(lambda+2)
    double Dparam;    ///< h^2/(96 mu_ij k_B pi^2), m^2 K; D_ij(T) = Dparam/T
    int fh;           ///< Feynman-Hibbs order, 1 or 2
};

inline PairParameters make_pair(const PureParameters& pi, const PureParameters& pj, double kij, double lij, int fh, double h_Js){
    if (fh != 1 && fh != 2){
        throw teqp::InvalidArgument("SAFT-VRQ Mie: the Feynman-Hibbs order must be 1 or 2");
    }
    PairParameters p;
    p.fh = fh;
    p.la = 3.0 + std::sqrt((pi.lambda_a - 3.0)*(pj.lambda_a - 3.0));
    p.lr = 3.0 + std::sqrt((pi.lambda_r - 3.0)*(pj.lambda_r - 3.0));
    p.sigma = 0.5*(pi.sigma_m + pj.sigma_m)*(1.0 - lij);
    p.eps = (1.0 - kij)*std::sqrt(std::pow(pi.sigma_m, 3)*std::pow(pj.sigma_m, 3))*std::sqrt(pi.epsilon_over_k*pj.epsilon_over_k)/std::pow(p.sigma, 3);
    p.C = p.lr/(p.lr - p.la)*std::pow(p.lr/p.la, p.la/(p.lr - p.la));
    p.Q1a = p.la*(p.la - 1.0); p.Q1r = p.lr*(p.lr - 1.0);
    p.Q2a = 0.5*p.la*(p.la - 1.0)*(p.la + 1.0)*(p.la + 2.0);
    p.Q2r = 0.5*p.lr*(p.lr - 1.0)*(p.lr + 1.0)*(p.lr + 2.0);
    double mu = pi.mass_kg*pj.mass_kg/(pi.mass_kg + pj.mass_kg);
    p.Dparam = h_Js*h_Js/(96.0*mu*constants::k_B*M_PI*M_PI);
    return p;
}

} // namespace teqp::SAFTVRQMie
