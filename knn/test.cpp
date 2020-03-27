/**
 * @file knn_main.cpp
 * @author Ryan Curtin
 *
 * Implementation of the kNN executable.  Allows some number of standard
 * options.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/tree/cover_tree.hpp>


#include <string>
#include <fstream>
#include <iostream>

#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/neighbor_search/unmap.hpp>
#include <mlpack/methods/neighbor_search/ns_model.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::util;

// Convenience typedef.
typedef NSModel<NearestNeighborSort> KNNModel;

// Information about the program itself.

int main()
{
    math::RandomSeed((size_t) std::time(NULL));

  // A user cannot specify both reference data and a model.
  
  // If the user specifies k but no output files, they should be warned.
  
  // If the user specifies output files but no k, they should be warned.
  
  // Sanity check on leaf size.
  const int lsInt = 5;

  // Sanity check on tau.

  const double tau = 2;


  // Sanity check on rho.
  const double rho = 0.2;

  // Sanity check on epsilon.
  const double epsilon = 3;

  // We either have to load the reference data, or we have to load the model.
  KNNModel* knn;

  const string algorithm =  "single_tree";
  //RequireParamInSet<string>("algorithm", { "naive", "single_tree", "dual_tree",
  //    "greedy" }, true, "unknown neighbor search algorithm");
  NeighborSearchMode searchMode = DUAL_TREE_MODE;

  if (algorithm == "naive")
    searchMode = NAIVE_MODE;
  else if (algorithm == "single_tree")
    searchMode = SINGLE_TREE_MODE;
  else if (algorithm == "dual_tree")
    searchMode = DUAL_TREE_MODE;
  else if (algorithm == "greedy")
    searchMode = GREEDY_SINGLE_TREE_MODE;


    knn = new KNNModel();

    // Get all the parameters.
    const string treeType = "kd";
    const bool randomBasis = true;

    KNNModel::TreeTypes tree = KNNModel::KD_TREE;
    //RequireParamInSet<string>("tree_type", { "kd", "cover", "r", "r-star",
    //    "ball", "x", "hilbert-r", "r-plus", "r-plus-plus", "spill", "vp", "rp",
    //    "max-rp", "ub", "oct" }, true, "unknown tree type");
    if (treeType == "kd")
      tree = KNNModel::KD_TREE;
    else if (treeType == "cover")
      tree = KNNModel::COVER_TREE;
    else if (treeType == "r")
      tree = KNNModel::R_TREE;
    else if (treeType == "r-star")
      tree = KNNModel::R_STAR_TREE;
    else if (treeType == "ball")
      tree = KNNModel::BALL_TREE;
    else if (treeType == "x")
      tree = KNNModel::X_TREE;
    else if (treeType == "hilbert-r")
      tree = KNNModel::HILBERT_R_TREE;
    else if (treeType == "r-plus")
      tree = KNNModel::R_PLUS_TREE;
    else if (treeType == "r-plus-plus")
      tree = KNNModel::R_PLUS_PLUS_TREE;
    else if (treeType == "spill")
      tree = KNNModel::SPILL_TREE;
    else if (treeType == "vp")
      tree = KNNModel::VP_TREE;
    else if (treeType == "rp")
      tree = KNNModel::RP_TREE;
    else if (treeType == "max-rp")
      tree = KNNModel::MAX_RP_TREE;
    else if (treeType == "ub")
      tree = KNNModel::UB_TREE;
    else if (treeType == "oct")
      tree = KNNModel::OCTREE;

    knn->TreeType() = tree;
    knn->RandomBasis() = randomBasis;
    knn->LeafSize() = size_t(lsInt);
    knn->Tau() = tau;
    knn->Rho() = rho;

    arma::mat referenceSet;
    data::Load("data2.csv", referenceSet, true);

    std::cout << "Loaded reference data from '"
        << "' ("
        << referenceSet.n_rows << " x " << referenceSet.n_cols << ")."
        << endl;

    knn->BuildModel(std::move(referenceSet), size_t(lsInt), searchMode,
        epsilon);



  // Perform search, if desired.

    const size_t k = 5;

    arma::mat queryData;
    //data::Load("data2.csv", queryData, true);

    // Sanity check on k value: must be greater than 0, must be less than or
    // equal to the number of reference points.  Since it is unsigned,
    // we only test the upper bound.
    
    // Sanity check on k value: must not be equal to the number of reference
    // points when query data has not been provided.
    
    // Now run the search.
    arma::Mat<size_t> neighbors;
    arma::mat distances;

      knn->Search(k, neighbors, distances);
    

    // Calculate the effective error, if desired.


    // Calculate the recall, if desired.


    // Save output.


  return 0;
}
