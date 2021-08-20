#include <Kokkos_Core.hpp>

#define EXPECT_EQ(a, b) std::cout << a << " == " << b << " ?" << std::endl;
#define EXPECT_LE(a, b) std::cout << a << " <= " << b << " ?" << std::endl;

#define scalar double

namespace Test {
struct TestLargeArgTag {};
struct TestRealErfcxTag {};
}

namespace Test {

template <class ExecSpace>
struct TestComplexErrorFunction {
  using ViewType = Kokkos::View<Kokkos::complex<scalar>*, ExecSpace>;
  using HostViewType =
      Kokkos::View<Kokkos::complex<scalar>*, Kokkos::HostSpace>;
  using DblViewType     = Kokkos::View<scalar*, ExecSpace>;
  using DblHostViewType = Kokkos::View<scalar*, Kokkos::HostSpace>;

  ViewType d_z, d_erf, d_erfcx;
  typename ViewType::HostMirror h_z, h_erf, h_erfcx;
  HostViewType h_ref_erf, h_ref_erfcx;

  DblViewType d_x, d_erfcx_dbl;
  typename DblViewType::HostMirror h_x, h_erfcx_dbl;
  DblHostViewType h_ref_erfcx_dbl;

  void testit() {
    using Kokkos::Experimental::infinity;

    d_z         = ViewType("d_z", 52);
    d_erf       = ViewType("d_erf", 52);
    d_erfcx     = ViewType("d_erfcx", 52);
    h_z         = Kokkos::create_mirror_view(d_z);
    h_erf       = Kokkos::create_mirror_view(d_erf);
    h_erfcx     = Kokkos::create_mirror_view(d_erfcx);
    h_ref_erf   = HostViewType("h_ref_erf", 52);
    h_ref_erfcx = HostViewType("h_ref_erfcx", 52);

    d_x             = DblViewType("d_x", 6);
    d_erfcx_dbl     = DblViewType("d_erfcx_dbl", 6);
    h_x             = Kokkos::create_mirror_view(d_x);
    h_erfcx_dbl     = Kokkos::create_mirror_view(d_erfcx_dbl);
    h_ref_erfcx_dbl = DblHostViewType("h_ref_erfcx_dbl", 6);

    // Generate test inputs
    // abs(z)<=2
    h_z(0)  = Kokkos::complex<scalar>(0.0011, 0);
    h_z(1)  = Kokkos::complex<scalar>(-0.0011, 0);
    h_z(2)  = Kokkos::complex<scalar>(1.4567, 0);
    h_z(3)  = Kokkos::complex<scalar>(-1.4567, 0);
    h_z(4)  = Kokkos::complex<scalar>(0, 0.0011);
    h_z(5)  = Kokkos::complex<scalar>(0, -0.0011);
    h_z(6)  = Kokkos::complex<scalar>(0, 1.4567);
    h_z(7)  = Kokkos::complex<scalar>(0, -1.4567);
    h_z(8)  = Kokkos::complex<scalar>(1.4567, 0.0011);
    h_z(9)  = Kokkos::complex<scalar>(1.4567, -0.0011);
    h_z(10) = Kokkos::complex<scalar>(-1.4567, 0.0011);
    h_z(11) = Kokkos::complex<scalar>(-1.4567, -0.0011);
    h_z(12) = Kokkos::complex<scalar>(1.4567, 0.5942);
    h_z(13) = Kokkos::complex<scalar>(1.4567, -0.5942);
    h_z(14) = Kokkos::complex<scalar>(-1.4567, 0.5942);
    h_z(15) = Kokkos::complex<scalar>(-1.4567, -0.5942);
    h_z(16) = Kokkos::complex<scalar>(0.0011, 0.5942);
    h_z(17) = Kokkos::complex<scalar>(0.0011, -0.5942);
    h_z(18) = Kokkos::complex<scalar>(-0.0011, 0.5942);
    h_z(19) = Kokkos::complex<scalar>(-0.0011, -0.5942);
    h_z(20) = Kokkos::complex<scalar>(0.0011, 0.0051);
    h_z(21) = Kokkos::complex<scalar>(0.0011, -0.0051);
    h_z(22) = Kokkos::complex<scalar>(-0.0011, 0.0051);
    h_z(23) = Kokkos::complex<scalar>(-0.0011, -0.0051);
    // abs(z)>2.0 and x>1
    h_z(24) = Kokkos::complex<scalar>(3.5, 0.0011);
    h_z(25) = Kokkos::complex<scalar>(3.5, -0.0011);
    h_z(26) = Kokkos::complex<scalar>(-3.5, 0.0011);
    h_z(27) = Kokkos::complex<scalar>(-3.5, -0.0011);
    h_z(28) = Kokkos::complex<scalar>(3.5, 9.7);
    h_z(29) = Kokkos::complex<scalar>(3.5, -9.7);
    h_z(30) = Kokkos::complex<scalar>(-3.5, 9.7);
    h_z(31) = Kokkos::complex<scalar>(-3.5, -9.7);
    h_z(32) = Kokkos::complex<scalar>(18.9, 9.7);
    h_z(33) = Kokkos::complex<scalar>(18.9, -9.7);
    h_z(34) = Kokkos::complex<scalar>(-18.9, 9.7);
    h_z(35) = Kokkos::complex<scalar>(-18.9, -9.7);
    // abs(z)>2.0 and 0<=x<=1 and abs(y)<6
    h_z(36) = Kokkos::complex<scalar>(0.85, 3.5);
    h_z(37) = Kokkos::complex<scalar>(0.85, -3.5);
    h_z(38) = Kokkos::complex<scalar>(-0.85, 3.5);
    h_z(39) = Kokkos::complex<scalar>(-0.85, -3.5);
    h_z(40) = Kokkos::complex<scalar>(0.0011, 3.5);
    h_z(41) = Kokkos::complex<scalar>(0.0011, -3.5);
    h_z(42) = Kokkos::complex<scalar>(-0.0011, 3.5);
    h_z(43) = Kokkos::complex<scalar>(-0.0011, -3.5);
    // abs(z)>2.0 and 0<=x<=1 and abs(y)>=6
    h_z(44) = Kokkos::complex<scalar>(0.85, 7.5);
    h_z(45) = Kokkos::complex<scalar>(0.85, -7.5);
    h_z(46) = Kokkos::complex<scalar>(-0.85, 7.5);
    h_z(47) = Kokkos::complex<scalar>(-0.85, -7.5);
    h_z(48) = Kokkos::complex<scalar>(0.85, 19.7);
    h_z(49) = Kokkos::complex<scalar>(0.85, -19.7);
    h_z(50) = Kokkos::complex<scalar>(-0.85, 19.7);
    h_z(51) = Kokkos::complex<scalar>(-0.85, -19.7);

    h_x(0) = -infinity<scalar>::value;
    h_x(1) = -1.2;
    h_x(2) = 0.0;
    h_x(3) = 1.2;
    h_x(4) = 10.5;
    h_x(5) = infinity<scalar>::value;

    Kokkos::deep_copy(d_z, h_z);
    Kokkos::deep_copy(d_x, h_x);

    // Call erf and erfcx functions
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 52), *this);
    Kokkos::fence();

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace, TestRealErfcxTag>(0, 1),
                         *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_erf, d_erf);
    Kokkos::deep_copy(h_erfcx, d_erfcx);
    Kokkos::deep_copy(h_erfcx_dbl, d_erfcx_dbl);

    // Reference values computed with Octave
    h_ref_erf(0) = Kokkos::complex<scalar>(0.001241216583181022, 0);
    h_ref_erf(1) = Kokkos::complex<scalar>(-0.001241216583181022, 0);
    h_ref_erf(2) = Kokkos::complex<scalar>(0.9606095744865353, 0);
    h_ref_erf(3) = Kokkos::complex<scalar>(-0.9606095744865353, 0);
    h_ref_erf(4) = Kokkos::complex<scalar>(0, 0.001241217584429469);
    h_ref_erf(5) = Kokkos::complex<scalar>(0, -0.001241217584429469);
    h_ref_erf(6) = Kokkos::complex<scalar>(0, 4.149756424218223);
    h_ref_erf(7) = Kokkos::complex<scalar>(0, -4.149756424218223);
    h_ref_erf(8) =
        Kokkos::complex<scalar>(0.960609812745064, 0.0001486911741082233);
    h_ref_erf(9) =
        Kokkos::complex<scalar>(0.960609812745064, -0.0001486911741082233);
    h_ref_erf(10) =
        Kokkos::complex<scalar>(-0.960609812745064, 0.0001486911741082233);
    h_ref_erf(11) =
        Kokkos::complex<scalar>(-0.960609812745064, -0.0001486911741082233);
    h_ref_erf(12) =
        Kokkos::complex<scalar>(1.02408827958197, 0.04828570635603527);
    h_ref_erf(13) =
        Kokkos::complex<scalar>(1.02408827958197, -0.04828570635603527);
    h_ref_erf(14) =
        Kokkos::complex<scalar>(-1.02408827958197, 0.04828570635603527);
    h_ref_erf(15) =
        Kokkos::complex<scalar>(-1.02408827958197, -0.04828570635603527);
    h_ref_erf(16) =
        Kokkos::complex<scalar>(0.001766791817179109, 0.7585038120712589);
    h_ref_erf(17) =
        Kokkos::complex<scalar>(0.001766791817179109, -0.7585038120712589);
    h_ref_erf(18) =
        Kokkos::complex<scalar>(-0.001766791817179109, 0.7585038120712589);
    h_ref_erf(19) =
        Kokkos::complex<scalar>(-0.001766791817179109, -0.7585038120712589);
    h_ref_erf(20) =
        Kokkos::complex<scalar>(0.001241248867618165, 0.005754776682713324);
    h_ref_erf(21) =
        Kokkos::complex<scalar>(0.001241248867618165, -0.005754776682713324);
    h_ref_erf(22) =
        Kokkos::complex<scalar>(-0.001241248867618165, 0.005754776682713324);
    h_ref_erf(23) =
        Kokkos::complex<scalar>(-0.001241248867618165, -0.005754776682713324);
    h_ref_erf(24) =
        Kokkos::complex<scalar>(0.9999992569244941, 5.939313159932013e-09);
    h_ref_erf(25) =
        Kokkos::complex<scalar>(0.9999992569244941, -5.939313159932013e-09);
    h_ref_erf(26) =
        Kokkos::complex<scalar>(-0.9999992569244941, 5.939313159932013e-09);
    h_ref_erf(27) =
        Kokkos::complex<scalar>(-0.9999992569244941, -5.939313159932013e-09);
    h_ref_erf(28) =
        Kokkos::complex<scalar>(-1.915595842013002e+34, 1.228821279117683e+32);
    h_ref_erf(29) =
        Kokkos::complex<scalar>(-1.915595842013002e+34, -1.228821279117683e+32);
    h_ref_erf(30) =
        Kokkos::complex<scalar>(1.915595842013002e+34, 1.228821279117683e+32);
    h_ref_erf(31) =
        Kokkos::complex<scalar>(1.915595842013002e+34, -1.228821279117683e+32);
    h_ref_erf(32) = Kokkos::complex<scalar>(1, 5.959897539826596e-117);
    h_ref_erf(33) = Kokkos::complex<scalar>(1, -5.959897539826596e-117);
    h_ref_erf(34) = Kokkos::complex<scalar>(-1, 5.959897539826596e-117);
    h_ref_erf(35) = Kokkos::complex<scalar>(-1, -5.959897539826596e-117);
    h_ref_erf(36) =
        Kokkos::complex<scalar>(-9211.077162784413, 13667.93825589455);
    h_ref_erf(37) =
        Kokkos::complex<scalar>(-9211.077162784413, -13667.93825589455);
    h_ref_erf(38) =
        Kokkos::complex<scalar>(9211.077162784413, 13667.93825589455);
    h_ref_erf(39) =
        Kokkos::complex<scalar>(9211.077162784413, -13667.93825589455);
    h_ref_erf(40) = Kokkos::complex<scalar>(259.38847811225, 35281.28906479814);
    h_ref_erf(41) =
        Kokkos::complex<scalar>(259.38847811225, -35281.28906479814);
    h_ref_erf(42) =
        Kokkos::complex<scalar>(-259.38847811225, 35281.28906479814);
    h_ref_erf(43) =
        Kokkos::complex<scalar>(-259.38847811225, -35281.28906479814);
    h_ref_erf(44) =
        Kokkos::complex<scalar>(6.752085728270252e+21, 9.809477366939276e+22);
    h_ref_erf(45) =
        Kokkos::complex<scalar>(6.752085728270252e+21, -9.809477366939276e+22);
    h_ref_erf(46) =
        Kokkos::complex<scalar>(-6.752085728270252e+21, 9.809477366939276e+22);
    h_ref_erf(47) =
        Kokkos::complex<scalar>(-6.752085728270252e+21, -9.809477366939276e+22);
    h_ref_erf(48) =
        Kokkos::complex<scalar>(4.37526734926942e+166, -2.16796709605852e+166);
    h_ref_erf(49) =
        Kokkos::complex<scalar>(4.37526734926942e+166, 2.16796709605852e+166);
    h_ref_erf(50) =
        Kokkos::complex<scalar>(-4.37526734926942e+166, -2.16796709605852e+166);
    h_ref_erf(51) =
        Kokkos::complex<scalar>(-4.37526734926942e+166, 2.16796709605852e+166);

    h_ref_erfcx(0) = Kokkos::complex<scalar>(0.9987599919156778, 0);
    h_ref_erfcx(1) = Kokkos::complex<scalar>(1.001242428085786, 0);
    h_ref_erfcx(2) = Kokkos::complex<scalar>(0.3288157848563544, 0);
    h_ref_erfcx(3) = Kokkos::complex<scalar>(16.36639786516915, 0);
    h_ref_erfcx(4) =
        Kokkos::complex<scalar>(0.999998790000732, -0.001241216082557101);
    h_ref_erfcx(5) =
        Kokkos::complex<scalar>(0.999998790000732, 0.001241216082557101);
    h_ref_erfcx(6) =
        Kokkos::complex<scalar>(0.1197948131677216, -0.4971192955307743);
    h_ref_erfcx(7) =
        Kokkos::complex<scalar>(0.1197948131677216, 0.4971192955307743);
    h_ref_erfcx(8) =
        Kokkos::complex<scalar>(0.3288156873503045, -0.0001874479383970247);
    h_ref_erfcx(9) =
        Kokkos::complex<scalar>(0.3288156873503045, 0.0001874479383970247);
    h_ref_erfcx(10) =
        Kokkos::complex<scalar>(16.36629202874158, -0.05369111060785572);
    h_ref_erfcx(11) =
        Kokkos::complex<scalar>(16.36629202874158, 0.05369111060785572);
    h_ref_erfcx(12) =
        Kokkos::complex<scalar>(0.3020886508118801, -0.09424097887578842);
    h_ref_erfcx(13) =
        Kokkos::complex<scalar>(0.3020886508118801, 0.09424097887578842);
    h_ref_erfcx(14) =
        Kokkos::complex<scalar>(-2.174707722732267, -11.67259764091796);
    h_ref_erfcx(15) =
        Kokkos::complex<scalar>(-2.174707722732267, 11.67259764091796);
    h_ref_erfcx(16) =
        Kokkos::complex<scalar>(0.7019810779371267, -0.5319516793968513);
    h_ref_erfcx(17) =
        Kokkos::complex<scalar>(0.7019810779371267, 0.5319516793968513);
    h_ref_erfcx(18) =
        Kokkos::complex<scalar>(0.7030703366403597, -0.5337884198542978);
    h_ref_erfcx(19) =
        Kokkos::complex<scalar>(0.7030703366403597, 0.5337884198542978);
    h_ref_erfcx(20) =
        Kokkos::complex<scalar>(0.9987340467266177, -0.005743428170378673);
    h_ref_erfcx(21) =
        Kokkos::complex<scalar>(0.9987340467266177, 0.005743428170378673);
    h_ref_erfcx(22) =
        Kokkos::complex<scalar>(1.001216353762532, -0.005765867613873103);
    h_ref_erfcx(23) =
        Kokkos::complex<scalar>(1.001216353762532, 0.005765867613873103);
    h_ref_erfcx(24) =
        Kokkos::complex<scalar>(0.1552936427089241, -4.545593205871305e-05);
    h_ref_erfcx(25) =
        Kokkos::complex<scalar>(0.1552936427089241, 4.545593205871305e-05);
    h_ref_erfcx(26) =
        Kokkos::complex<scalar>(417949.5262869648, -3218.276197742372);
    h_ref_erfcx(27) =
        Kokkos::complex<scalar>(417949.5262869648, 3218.276197742372);
    h_ref_erfcx(28) =
        Kokkos::complex<scalar>(0.01879467905925653, -0.0515934271478583);
    h_ref_erfcx(29) =
        Kokkos::complex<scalar>(0.01879467905925653, 0.0515934271478583);
    h_ref_erfcx(30) =
        Kokkos::complex<scalar>(-0.01879467905925653, -0.0515934271478583);
    h_ref_erfcx(31) =
        Kokkos::complex<scalar>(-0.01879467905925653, 0.0515934271478583);
    h_ref_erfcx(32) =
        Kokkos::complex<scalar>(0.02362328821805, -0.01209735551897239);
    h_ref_erfcx(33) =
        Kokkos::complex<scalar>(0.02362328821805, 0.01209735551897239);
    h_ref_erfcx(34) = Kokkos::complex<scalar>(-2.304726099084567e+114,
                                              -2.942443198107089e+114);
    h_ref_erfcx(35) = Kokkos::complex<scalar>(-2.304726099084567e+114,
                                              2.942443198107089e+114);
    h_ref_erfcx(36) =
        Kokkos::complex<scalar>(0.04174017523145063, -0.1569865319886248);
    h_ref_erfcx(37) =
        Kokkos::complex<scalar>(0.04174017523145063, 0.1569865319886248);
    h_ref_erfcx(38) =
        Kokkos::complex<scalar>(-0.04172154858670504, -0.156980085534407);
    h_ref_erfcx(39) =
        Kokkos::complex<scalar>(-0.04172154858670504, 0.156980085534407);
    h_ref_erfcx(40) =
        Kokkos::complex<scalar>(6.355803055239174e-05, -0.1688298297427782);
    h_ref_erfcx(41) =
        Kokkos::complex<scalar>(6.355803055239174e-05, 0.1688298297427782);
    h_ref_erfcx(42) =
        Kokkos::complex<scalar>(-5.398806789669434e-05, -0.168829903432947);
    h_ref_erfcx(43) =
        Kokkos::complex<scalar>(-5.398806789669434e-05, 0.168829903432947);
    h_ref_erfcx(44) =
        Kokkos::complex<scalar>(0.008645103282302355, -0.07490521021566741);
    h_ref_erfcx(45) =
        Kokkos::complex<scalar>(0.008645103282302355, 0.07490521021566741);
    h_ref_erfcx(46) =
        Kokkos::complex<scalar>(-0.008645103282302355, -0.07490521021566741);
    h_ref_erfcx(47) =
        Kokkos::complex<scalar>(-0.008645103282302355, 0.07490521021566741);
    h_ref_erfcx(48) =
        Kokkos::complex<scalar>(0.001238176693606428, -0.02862247416909219);
    h_ref_erfcx(49) =
        Kokkos::complex<scalar>(0.001238176693606428, 0.02862247416909219);
    h_ref_erfcx(50) =
        Kokkos::complex<scalar>(-0.001238176693606428, -0.02862247416909219);
    h_ref_erfcx(51) =
        Kokkos::complex<scalar>(-0.001238176693606428, 0.02862247416909219);

    h_ref_erfcx_dbl(0) = infinity<scalar>::value;
    h_ref_erfcx_dbl(1) = 8.062854217063865e+00;
    h_ref_erfcx_dbl(2) = 1.0;
    h_ref_erfcx_dbl(3) = 3.785374169292397e-01;
    h_ref_erfcx_dbl(4) = 5.349189974656411e-02;
    h_ref_erfcx_dbl(5) = 0.0;

    for (int i = 0; i < 52; i++) {
      EXPECT_LE(Kokkos::abs(h_erf(i) - h_ref_erf(i)),
                Kokkos::abs(h_ref_erf(i)) * 1e-13);
    }

    for (int i = 0; i < 52; i++) {
      EXPECT_LE(Kokkos::abs(h_erfcx(i) - h_ref_erfcx(i)),
                Kokkos::abs(h_ref_erfcx(i)) * 1e-13);
    }

    EXPECT_EQ(h_erfcx_dbl(0), h_ref_erfcx_dbl(0));
    EXPECT_EQ(h_erfcx_dbl(5), h_ref_erfcx_dbl(5));
    for (int i = 1; i < 5; i++) {
      EXPECT_LE(std::abs(h_erfcx_dbl(i) - h_ref_erfcx_dbl(i)),
                std::abs(h_ref_erfcx_dbl(i)) * 1e-13);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const {
    d_erf(i)   = Kokkos::Experimental::erf(d_z(i));
    d_erfcx(i) = Kokkos::Experimental::erfcx(d_z(i));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TestRealErfcxTag&, const int& /*i*/) const {
    d_erfcx_dbl(0) = Kokkos::Experimental::erfcx(d_x(0));
    d_erfcx_dbl(1) = Kokkos::Experimental::erfcx(d_x(1));
    d_erfcx_dbl(2) = Kokkos::Experimental::erfcx(d_x(2));
    d_erfcx_dbl(3) = Kokkos::Experimental::erfcx(d_x(3));
    d_erfcx_dbl(4) = Kokkos::Experimental::erfcx(d_x(4));
    d_erfcx_dbl(5) = Kokkos::Experimental::erfcx(d_x(5));
  }
};


}  // namespace Test

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        using TEST_EXECSPACE = Kokkos::DefaultExecutionSpace;
        Test::TestComplexErrorFunction<TEST_EXECSPACE> test;
        test.testit();

        printf("Done \n");
    }
    Kokkos::finalize();
    return 0;
}
