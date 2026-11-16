#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
using Catch::Approx;

#include "teqp/models/multifluid/multifluid_excess.hpp"
#include "teqp/derivs.hpp"
#include "teqp/cpp/teqpcpp.hpp"

#include "test_common.in"

using namespace teqp;
using teqp::multifluid::multifluid_excess::MultifluidPlusExcess;

/// Methane + ethane: both are in teqp's fluid library and in the SAFT-VR Mie library
static nlohmann::json hybrid_spec(){
    return {
        {"multifluid", {{"components", {"Methane", "Ethane"}}, {"root", FLUIDDATAPATH}, {"BIP", ""}, {"departure", ""}}},
        {"excess", {{"kind", "SAFT-VR-Mie"}, {"model", {{"names", {"Methane", "Ethane"}}}}}}
    };
}

TEST_CASE("multifluid-excess: the pure-fluid limits are the multifluid's to the bit", "[multifluid-excess]"){
    const MultifluidPlusExcess hybrid(hybrid_spec());
    const auto mf = multifluidfactory(hybrid_spec().at("multifluid"));
    for (int i = 0; i < 2; ++i){
        Eigen::ArrayXd z = Eigen::ArrayXd::Zero(2); z[i] = 1.0;
        for (double T : {150.0, 300.0, 500.0}){
            for (double rho : {1.0, 1e3, 1e4, 2e4}){
                CAPTURE(i, T, rho);
                CHECK(hybrid.alphar(T, rho, z) == mf.alphar(T, rho, z));
                CHECK(hybrid.alphar_excess(T, rho, z) == 0.0);
            }
        }
    }
}

TEST_CASE("multifluid-excess: the excess against single-component SAFT-VR Mie models", "[multifluid-excess]"){
    const MultifluidPlusExcess hybrid(hybrid_spec());
    const auto mix = SAFTVRMie::SAFTVRMiefactory({{"names", {"Methane", "Ethane"}}});
    const auto pure0 = SAFTVRMie::SAFTVRMiefactory({{"names", {"Methane"}}});
    const auto pure1 = SAFTVRMie::SAFTVRMiefactory({{"names", {"Ethane"}}});
    Eigen::ArrayXd one(1); one << 1.0;
    for (double x0 : {0.1, 0.5, 0.9}){
        Eigen::ArrayXd z(2); z << x0, 1.0-x0;
        for (double T : {150.0, 300.0}){
            for (double rho : {1.0, 1e3, 1e4}){
                CAPTURE(x0, T, rho);
                double expected = mix.alphar(T, rho, z) - x0*pure0.alphar(T, rho, one) - (1.0-x0)*pure1.alphar(T, rho, one);
                CHECK(hybrid.alphar_excess(T, rho, z) == Approx(expected).epsilon(1e-13));
                CHECK(std::abs(hybrid.alphar_excess(T, rho, z)) > 0.0); // a real mixture has a nonzero excess
            }
        }
    }
}

TEST_CASE("multifluid-excess: autodiff derivatives against complex step and finite differences", "[multifluid-excess]"){
    const MultifluidPlusExcess hybrid(hybrid_spec());
    using tdx = TDXDerivatives<MultifluidPlusExcess>;
    using id = IsochoricDerivatives<MultifluidPlusExcess>;
    Eigen::ArrayXd z(2); z << 0.4, 0.6;
    const std::vector<std::pair<double, double>> states = {{200.0, 500.0}, {200.0, 1e3}, {300.0, 500.0}, {300.0, 1e4}};
    for (auto [T, rho] : states){
        CAPTURE(T, rho);
        CHECK(tdx::get_Ar01<ADBackends::autodiff>(hybrid, T, rho, z)
              == Approx(tdx::get_Ar01<ADBackends::complex_step>(hybrid, T, rho, z)).epsilon(1e-12));
        // Ar10 reference uses multicomplex: get_Arxy<1,0,complex_step> calls w.alpha, which alphar-only models lack
        CHECK(tdx::get_Ar10<ADBackends::autodiff>(hybrid, T, rho, z)
              == Approx(tdx::get_Ar10<ADBackends::multicomplex>(hybrid, T, rho, z)).epsilon(1e-12));
        // a central difference of alphar in rho is roundoff-limited at ~1e-4 here; multicomplex is exact
        CHECK(tdx::get_Ar02<ADBackends::autodiff>(hybrid, T, rho, z)
              == Approx(tdx::get_Ar02<ADBackends::multicomplex>(hybrid, T, rho, z)).epsilon(1e-12));
        CHECK(tdx::get_Ar11<ADBackends::autodiff>(hybrid, T, rho, z)
              == Approx(tdx::get_Ar11<ADBackends::multicomplex>(hybrid, T, rho, z)).epsilon(1e-12));
        // ln fugacity coefficients: d(n alphar)/dn_i by central difference in the mole numbers
        double Z = 1.0 + tdx::get_Ar01(hybrid, T, rho, z);
        REQUIRE(Z > 0);
        Eigen::ArrayXd rhovec = rho*z;
        auto lnphi = id::get_ln_fugacity_coefficients(hybrid, T, rhovec);
        for (int i = 0; i < 2; ++i){
            // dn = 1e-5 balances truncation against roundoff of n*alphar; at 1e-6 roundoff is ~5e-10, the tolerance at ln phi ~0.005
            double dn = 1e-5;
            // the multifluid's R depends on composition; teqp's ln phi differentiates R(x) rho alphar
            auto nalphar = [&](double ni_shift){
                Eigen::ArrayXd n = z; n[i] += ni_shift;   // V = 1/rho fixed, total moles change
                double ntot = n.sum();
                Eigen::ArrayXd x = (n/ntot).eval();
                return ntot*hybrid.alphar(T, ntot*rho, x)*hybrid.R(x)/hybrid.R(z);
            };
            double dnadn = (nalphar(dn) - nalphar(-dn))/(2*dn);
            CAPTURE(i);
            CHECK(lnphi[i] == Approx(dnadn - std::log(Z)).epsilon(1e-7));
        }
    }
}

TEST_CASE("multifluid-excess: built by kind through the abstract interface", "[multifluid-excess]"){
    const MultifluidPlusExcess hybrid(hybrid_spec());
    auto am = cppinterface::make_model({{"kind", "multifluid-excess"}, {"model", hybrid_spec()}});
    Eigen::ArrayXd z(2); z << 0.3, 0.7;
    CHECK(am->get_Ar00(250.0, 5000.0, z) == hybrid.alphar(250.0, 5000.0, z));
    CHECK(am->get_Ar01(250.0, 5000.0, z) == Approx(TDXDerivatives<MultifluidPlusExcess>::get_Ar01(hybrid, 250.0, 5000.0, z)).epsilon(1e-14));
}

TEST_CASE("multifluid-excess: invalid specifications raise", "[multifluid-excess]"){
    auto wrong_kind = hybrid_spec(); wrong_kind["excess"]["kind"] = "PCSAFT";
    CHECK_THROWS_AS(MultifluidPlusExcess(wrong_kind), teqp::InvalidArgument);
    CHECK_THROWS_MATCHES(MultifluidPlusExcess(wrong_kind), teqp::InvalidArgument,
                         Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("must be of kind SAFT-VR-Mie")));
    auto wrong_size = hybrid_spec(); wrong_size["excess"]["model"]["names"] = {"Methane"};
    CHECK_THROWS_AS(MultifluidPlusExcess(wrong_size), teqp::InvalidArgument);
    CHECK_THROWS_MATCHES(MultifluidPlusExcess(wrong_size), teqp::InvalidArgument,
                         Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("components")));
}

