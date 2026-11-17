#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
using Catch::Approx;

#include <complex>

#include "teqp/models/multifluid/multifluid_excess.hpp"
#include "teqp/models/saftvrqmie.hpp"
#include "teqp/json_tools.hpp"
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

TEST_CASE("multifluid-excess: polar SAFT-VR Mie input raises instead of being dropped", "[multifluid-excess]"){
    auto coeffs_spec = [](const nlohmann::json& extra){
        auto spec = hybrid_spec();
        nlohmann::json c0 = {{"name", "Methane"}, {"m", 1.0}, {"sigma_m", 3.7412e-10}, {"epsilon_over_k", 153.36},
                             {"lambda_r", 12.650}, {"lambda_a", 6.0}, {"BibTeXKey", "Lafitte-JCP-2001"}};
        nlohmann::json c1 = {{"name", "Ethane"}, {"m", 1.4373}, {"sigma_m", 3.7257e-10}, {"epsilon_over_k", 206.12},
                             {"lambda_r", 12.400}, {"lambda_a", 6.0}, {"BibTeXKey", "Lafitte-JCP-2001"}};
        c1.update(extra);
        spec["excess"]["model"] = {{"coeffs", {c0, c1}}};
        return spec;
    };
    // the same fluids given by coefficients build, and give the excess of the "names" form
    const MultifluidPlusExcess by_names(hybrid_spec()), by_coeffs(coeffs_spec(nlohmann::json::object()));
    Eigen::ArrayXd z(2); z << 0.3, 0.7;
    CHECK(by_coeffs.alphar_excess(250.0, 5000.0, z) == by_names.alphar_excess(250.0, 5000.0, z));
    for (const auto& extra : {nlohmann::json{{"mu_D", 1.0}}, nlohmann::json{{"Q_DA", 1.0}},
                              nlohmann::json{{"(mu^*)^2", 0.5}, {"nmu", 1.0}}, nlohmann::json{{"Q_Cm2", 1e-40}}}){
        CAPTURE(extra.dump());
        CHECK_THROWS_MATCHES(MultifluidPlusExcess(coeffs_spec(extra)), teqp::InvalidArgument,
                             Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("nonpolar")));
    }
    auto with_polar_model = hybrid_spec(); with_polar_model["excess"]["model"]["polar_model"] = "GrossVrabec";
    CHECK_THROWS_MATCHES(MultifluidPlusExcess(with_polar_model), teqp::InvalidArgument,
                         Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("nonpolar")));
}

TEST_CASE("multifluid-excess: a VLE isotherm traces from the pure-fluid saturation state", "[multifluid-excess]"){
    auto am = cppinterface::make_model({{"kind", "multifluid-excess"}, {"model", hybrid_spec()}});
    auto mf = cppinterface::make_model({{"kind", "multifluid"}, {"model", hybrid_spec().at("multifluid")}});
    const double T = 230.0;  // below Tc(ethane) 305.3 K, above Tc(methane) 190.6 K
    Eigen::ArrayXd z1(2); z1 << 0.0, 1.0;
    // The abstract interface has no superancillary accessor, so both models start from the same guesses
    auto rhoLV_mf = mf->pure_VLE_T(T, 15000.0, 200.0, 50, z1);
    auto rhoLV = am->pure_VLE_T(T, 15000.0, 200.0, 50, z1);
    for (int i = 0; i < 2; ++i){
        REQUIRE(std::isfinite(rhoLV_mf[i]));
        REQUIRE(std::isfinite(rhoLV[i]));
        CHECK(rhoLV[i] == Approx(rhoLV_mf[i]).epsilon(1e-10));
    }
    REQUIRE(rhoLV[0] > rhoLV[1]);
    Eigen::ArrayXd rhovecL(2), rhovecV(2); rhovecL << 0.0, rhoLV[0]; rhovecV << 0.0, rhoLV[1];
    TVLEOptions opt; opt.p_termination = 1e8; opt.crit_termination = 1e-4; opt.calc_criticality = true;
    auto trace = am->trace_VLE_isotherm_binary(T, rhovecL, rhovecV, opt);
    REQUIRE(trace.size() > 10);
    double max_xL0 = 0.0;
    // The multifluid's R depends on composition (methane 8.31451, ethane 8.314472). The tracer solves
    // with R(x) of the liquid for both phases but stores pV with R(y), so pL and pV differ by up to
    // ~3e-6 relative, in the plain multifluid as well. pL is compared with pV recomputed with R(x).
    for (const auto& pt : trace){
        double pL = pt.at("pL / Pa").get<double>();
        auto rhovecLpt = pt.at("rhoL / mol/m^3").get<std::vector<double>>();
        auto rhovecVpt = pt.at("rhoV / mol/m^3").get<std::vector<double>>();
        Eigen::ArrayXd rL = Eigen::Map<Eigen::ArrayXd>(rhovecLpt.data(), 2);
        Eigen::ArrayXd rV = Eigen::Map<Eigen::ArrayXd>(rhovecVpt.data(), 2);
        double pV_RxL = rV.sum()*am->get_R((rL/rL.sum()).eval())*T + am->get_pr(T, rV);
        CAPTURE(pL, pV_RxL);
        CHECK(std::isfinite(pL));
        CHECK(pL == Approx(pV_RxL).epsilon(1e-10));
        max_xL0 = std::max(max_xL0, pt.at("xL_0 / mole frac.").get<double>());
    }
    CAPTURE(trace.size(), max_xL0);
    CHECK(max_xL0 > 0.05);
}

// ---------------------------------------------------------------------------------------------------
// He-Ne: multifluid from teqp's fluid library plus a SAFT-VRQ Mie (Feynman-Hibbs order 1) excess
// ---------------------------------------------------------------------------------------------------

/// The SAFT-VRQ Mie block of one thermopack 2.2.3 reference system (src/tests/resources/saftvrqmie_thermopack_2.2.3.json),
/// with thermopack's masses and Planck constant. The same few lines as model_spec in catch_test_SAFTVRQMie.cxx,
/// duplicated so that the two test files stay independent.
static nlohmann::json saftvrq_block(const std::string& name, int fh){
    static const nlohmann::json ref = teqp::load_a_JSON_file("../src/tests/resources/saftvrqmie_thermopack_2.2.3.json");
    auto mass_kg = [](const std::string& tpname){  // saftvrmie_datadb.f90 of thermopack 2.2.3
        if (tpname == "HE") return 6.6464764e-27;
        if (tpname == "NE") return 3.3509177e-26;
        throw std::invalid_argument(tpname);
    };
    for (const auto& sys : ref["systems"]){
        if (sys["name"] != name || sys["feynman_hibbs_order"] != fh){ continue; }
        nlohmann::json coeffs = nlohmann::json::array();
        for (auto i = 0U; i < sys["components"].size(); ++i){
            const auto& p = sys["parameters"]["pure"][i];
            coeffs.push_back({{"name", sys["components"][i]}, {"sigma_m", p["sigma_m"]}, {"epsilon_over_k", p["epsilon_over_k_K"]},
                              {"lambda_a", p["lambda_a"]}, {"lambda_r", p["lambda_r"]}, {"mass_kg", mass_kg(sys["components"][i])}});
        }
        return {{"feynman_hibbs_order", sys["feynman_hibbs_order"]}, {"coeffs", coeffs},
                {"kmat", sys["parameters"]["kij"]}, {"lmat", sys["parameters"]["lij"]},
                {"Planck_constant / J s", ref["constants_thermopack"]["h_J_s"]}};
    }
    throw std::invalid_argument(name);
}

/// teqp's mixture_binary_pairs.json has no Helium-Neon entry, so the reducing parameters are given: all 1, no departure
static nlohmann::json heNe_spec(){
    nlohmann::json BIP = nlohmann::json::array();
    BIP.push_back({{"BibTeX", ""}, {"CAS1", "7440-59-7"}, {"CAS2", "7440-01-9"}, {"Name1", "Helium"}, {"Name2", "Neon"},
                   {"betaT", 1.0}, {"gammaT", 1.0}, {"betaV", 1.0}, {"gammaV", 1.0}, {"F", 0.0}});
    return {
        {"multifluid", {{"components", {"Helium", "Neon"}}, {"root", FLUIDDATAPATH}, {"BIP", BIP}, {"departure", ""}}},
        {"excess", {{"kind", "SAFT-VRQ-Mie"}, {"model", saftvrq_block("He-Ne x_He=0.5", 1)}}}
    };
}

/// Gradient of Psir = alphar R T rho w.r.t. the molar concentrations by complex step, one component at a time.
/// teqp's build_Psir_gradient_complex_step does not compile, so it is written here (as in catch_test_SAFTVRQMie.cxx).
template<typename Model>
static Eigen::ArrayXd Psir_gradient_complex_step(const Model& model, double T, const Eigen::ArrayXd& rhovec){
    const double h = 1e-100;
    const Eigen::ArrayXd xreal = rhovec/rhovec.sum();
    const double R = model.R(xreal);
    Eigen::ArrayXd g(rhovec.size());
    for (auto i = 0; i < rhovec.size(); ++i){
        Eigen::ArrayX<std::complex<double>> r = rhovec.cast<std::complex<double>>();
        r[i] += std::complex<double>(0.0, h);
        const std::complex<double> rhotot = r.sum();
        const Eigen::ArrayX<std::complex<double>> x = (r/rhotot).eval();
        g[i] = (model.alphar(T, rhotot, x)*R*T*rhotot).imag()/h;
    }
    return g;
}

TEST_CASE("multifluid-excess SAFT-VRQ Mie: the He-Ne pure-fluid limits are the multifluid's to the bit", "[multifluid-excess]"){
    const MultifluidPlusExcess hybrid(heNe_spec());
    const auto mf = multifluidfactory(heNe_spec().at("multifluid"));
    for (int i = 0; i < 2; ++i){
        Eigen::ArrayXd z = Eigen::ArrayXd::Zero(2); z[i] = 1.0;
        for (double T : {20.0, 30.0, 50.0, 150.0}){
            for (double rho : {1.0, 1e3, 1e4, 4e4}){
                CAPTURE(i, T, rho);
                CHECK(hybrid.alphar(T, rho, z) == mf.alphar(T, rho, z));
                CHECK(hybrid.alphar_excess(T, rho, z) == 0.0);
            }
        }
    }
}

TEST_CASE("multifluid-excess SAFT-VRQ Mie: the He-Ne excess against single-component SAFT-VRQ Mie models", "[multifluid-excess]"){
    const MultifluidPlusExcess hybrid(heNe_spec());
    const auto mix = SAFTVRQMie::SAFTVRQMiefactory(saftvrq_block("He-Ne x_He=0.5", 1));
    const auto pure0 = SAFTVRQMie::SAFTVRQMiefactory(saftvrq_block("He", 1));
    const auto pure1 = SAFTVRQMie::SAFTVRQMiefactory(saftvrq_block("Ne", 1));
    Eigen::ArrayXd one(1); one << 1.0;
    for (double x0 : {0.05, 0.5, 0.9}){
        Eigen::ArrayXd z(2); z << x0, 1.0-x0;
        for (double T : {20.0, 30.0, 50.0, 150.0}){
            for (double rho : {1.0, 1e3, 1e4}){
                CAPTURE(x0, T, rho);
                double expected = mix.alphar(T, rho, z) - x0*pure0.alphar(T, rho, one) - (1.0-x0)*pure1.alphar(T, rho, one);
                CHECK(hybrid.alphar_excess(T, rho, z) == Approx(expected).epsilon(1e-13));
                CHECK(std::abs(hybrid.alphar_excess(T, rho, z)) > 0.0);
            }
        }
    }
}

TEST_CASE("multifluid-excess SAFT-VRQ Mie: He-Ne autodiff derivatives against complex step and multicomplex", "[multifluid-excess]"){
    const MultifluidPlusExcess hybrid(heNe_spec());
    using tdx = TDXDerivatives<MultifluidPlusExcess>;
    using id = IsochoricDerivatives<MultifluidPlusExcess>;
    for (double x0 : {0.05, 0.5}){
        Eigen::ArrayXd z(2); z << x0, 1.0-x0;
        for (double T : {20.0, 30.0, 50.0, 150.0}){
            for (double rho : {500.0, 1e4, 3e4}){
                CAPTURE(x0, T, rho);
                CHECK(tdx::get_Ar01<ADBackends::autodiff>(hybrid, T, rho, z)
                      == Approx(tdx::get_Ar01<ADBackends::complex_step>(hybrid, T, rho, z)).epsilon(1e-12));
                // Ar10 reference uses multicomplex: get_Arxy<1,0,complex_step> calls w.alpha, which alphar-only models lack
                CHECK(tdx::get_Ar10<ADBackends::autodiff>(hybrid, T, rho, z)
                      == Approx(tdx::get_Ar10<ADBackends::multicomplex>(hybrid, T, rho, z)).epsilon(1e-12));
                const Eigen::ArrayXd rhovec = (rho*z).eval();
                const Eigen::ArrayXd g_ad = id::build_Psir_gradient_autodiff(hybrid, T, rhovec);
                const Eigen::ArrayXd g_cs = Psir_gradient_complex_step(hybrid, T, rhovec);
                for (auto i = 0; i < 2; ++i){
                    CAPTURE(i, g_ad[i], g_cs[i]);
                    CHECK(g_ad[i] == Approx(g_cs[i]).epsilon(1e-12));
                }
            }
        }
    }
}

TEST_CASE("multifluid-excess SAFT-VRQ Mie: a He-Ne isotherm at 30 K traces from pure neon", "[multifluid-excess]"){
    auto am = cppinterface::make_model({{"kind", "multifluid-excess"}, {"model", heNe_spec()}});
    auto mf = cppinterface::make_model({{"kind", "multifluid"}, {"model", heNe_spec().at("multifluid")}});
    const double T = 30.0;  // above Tc(helium) 5.2 K, below Tc(neon) 44.4 K
    Eigen::ArrayXd z1(2); z1 << 0.0, 1.0;
    auto rhoLV_mf = mf->pure_VLE_T(T, 58000.0, 1000.0, 50, z1);
    auto rhoLV = am->pure_VLE_T(T, 58000.0, 1000.0, 50, z1);
    for (int i = 0; i < 2; ++i){
        REQUIRE(std::isfinite(rhoLV_mf[i]));
        REQUIRE(std::isfinite(rhoLV[i]));
        CHECK(rhoLV[i] == Approx(rhoLV_mf[i]).epsilon(1e-10));
    }
    REQUIRE(rhoLV[0] > rhoLV[1]);
    Eigen::ArrayXd rhovecL(2), rhovecV(2); rhovecL << 0.0, rhoLV[0]; rhovecV << 0.0, rhoLV[1];
    TVLEOptions opt; opt.p_termination = 1e8; opt.crit_termination = 1e-4; opt.calc_criticality = true;
    auto trace = am->trace_VLE_isotherm_binary(T, rhovecL, rhovecV, opt);
    REQUIRE(trace.size() > 10);
    double max_yV0 = 0.0;
    for (const auto& pt : trace){
        double pL = pt.at("pL / Pa").get<double>();
        auto rhovecLpt = pt.at("rhoL / mol/m^3").get<std::vector<double>>();
        auto rhovecVpt = pt.at("rhoV / mol/m^3").get<std::vector<double>>();
        Eigen::ArrayXd rL = Eigen::Map<Eigen::ArrayXd>(rhovecLpt.data(), 2);
        Eigen::ArrayXd rV = Eigen::Map<Eigen::ArrayXd>(rhovecVpt.data(), 2);
        double pV_RxL = rV.sum()*am->get_R((rL/rL.sum()).eval())*T + am->get_pr(T, rV);
        CAPTURE(pL, pV_RxL);
        CHECK(std::isfinite(pL));
        CHECK(pL == Approx(pV_RxL).epsilon(1e-10));
        max_yV0 = std::max(max_yV0, pt.at("xV_0 / mole frac.").get<double>());
    }
    CAPTURE(trace.size(), max_yV0);
    // measured: max liquid x_He 8.15e-4, max vapour y_He 0.99929, ends on p_termination at 95.46 MPa (87 points)
    CHECK(max_yV0 > 0.5);
}
