/**
 * @file pca_main.cpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Main executable to run PCA.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/pca/pca.hpp>
#include <mlpack/methods/pca/decomposition_policies/exact_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/quic_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/randomized_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/randomized_block_krylov_method.hpp>

using namespace mlpack;
using namespace mlpack::pca;
using namespace mlpack::util;
using namespace std;

// Document program.


//! Run RunPCA on the specified dataset with the given decomposition method.
template<typename DecompositionPolicy>
void RunPCA(arma::mat& dataset,
            const size_t newDimension,
            const bool scale,
            const double varToRetain)
{
  PCA<DecompositionPolicy> p(scale);


  double varRetained;

    varRetained = p.Apply(dataset, newDimension);


  std::cout << (varRetained * 100) << "% of variance retained (" <<
      dataset.n_rows << " dimensions)." << endl;
}

int main()
{
  // Load input dataset.
  arma::mat dataset;
  data::Load("data2.csv", dataset, true);

  // Issue a warning if the user did not specify an output file.
    size_t newDimension =dataset.n_rows ;

  // Get the options for running PCA.
  const bool scale = true;
  const double varToRetain =3;
  const string decompositionMethod ="exact";

  // Perform PCA.
  if (decompositionMethod == "exact")
  {
    RunPCA<ExactSVDPolicy>(dataset, newDimension, scale, varToRetain);
  }
  else if (decompositionMethod == "randomized")
  {
    RunPCA<RandomizedSVDPolicy>(dataset, newDimension, scale, varToRetain);
  }
  else if (decompositionMethod == "randomized-block-krylov")
  {
    RunPCA<RandomizedBlockKrylovSVDPolicy>(dataset, newDimension, scale,
        varToRetain);
  }
  else if (decompositionMethod == "quic")
  {
    RunPCA<QUICSVDPolicy>(dataset, newDimension, scale, varToRetain);
  }

  // Now save the results.

}

