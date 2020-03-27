#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include<omp.h>
using namespace mlpack;
using namespace mlpack::neighbor; // NeighborSearch and NearestNeighborSort
using namespace mlpack::metric; // ManhattanDistance
using namespace std;
int main(){



  arma::mat dataset;

  
  data::Load("test_data_3_1000.csv", dataset);
  KNN exact(dataset);
  arma::Mat<size_t> neighborsExact;
  arma::mat distancesExact;
  double t=omp_get_wtime();
  exact.Search(15, neighborsExact, distancesExact);
  t=omp_get_wtime()-t;
  cout<<t<<endl;

  KNN aknn(dataset, DUAL_TREE_MODE, 0.05);
  arma::Mat<size_t> neighborsApprox;
  arma::mat distancesApprox;
  aknn.Search(15, neighborsApprox, distancesApprox);
  
  return 0;
}
