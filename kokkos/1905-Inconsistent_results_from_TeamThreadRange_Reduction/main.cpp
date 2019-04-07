#include<Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  int errors;
  {
     int N = (argc>1) ? atoi(argv[1]) : 10;
     int M = (argc>2) ? atoi(argv[2]) : 64;
     int R = (argc>3) ? atoi(argv[3]) : 10;
    
     Kokkos::View<double*> results1("r1",N);
     Kokkos::View<double*> results2("r2",N);
     Kokkos::View<double**> data("d",N,M);
     Kokkos::deep_copy(data,1);
     Kokkos::parallel_for(Kokkos::TeamPolicy<>(N,R), KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team) {
       const int i = team.league_rank();
       double s;
       Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,M), [&] (const int j, double& update) {
         update += data(i,j) + 1000*i + j;
       },s);
       results2(i) = s;
       Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,M), [&] (const int j, double& update) {
         update += data(i,j) + 1000*i + j;
       },results1(i));
       team.team_barrier();
     });


     Kokkos::parallel_reduce(N, KOKKOS_LAMBDA (const int i, int& lerror) {
       double expected = 1.0*(M*(1000*i+1)+M*(M-1)/2);
       printf("%i %lf %lf %lf\n",i,results1(i),results2(i),expected);
       if((expected != results1(i)) || (expected != results1(i)))
         lerror++;
     },errors);
  }
  Kokkos::finalize();
  printf("Errors: %i\n",errors);
  if(errors > 0) return 1;
}
