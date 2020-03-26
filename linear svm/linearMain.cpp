/**
 * @file linear_svm_main.cpp
 * @author Yashwant Singh Parihar
 *
 * Main executable for linear svm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core.hpp>


#include <mlpack/methods/linear_svm/linear_svm.hpp>

#include <ensmallen.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::svm;
using namespace mlpack::util;


class LinearSVMModel
{
 public:
  arma::Col<size_t> mappings;
  LinearSVM<> svm;

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(mappings);
    ar & BOOST_SERIALIZATION_NVP(svm);
  }
};


// Model loading/saving.
int main()
{

  // Collect command-line options.
  const double lambda =2;
  const double delta = 10;
  const string optimizerType ="lbfgs";
  const double tolerance =10;
  const bool intercept = 20;
  const size_t epochs = 20;
  const size_t maxIterations =20;
;

  // One of training and input_model must be specified.




  // Delta must be positive.

  // These are the matrices we might use.
  arma::mat trainingSet;
  arma::Row<size_t> labels;
  arma::Row<size_t> rawLabels;
  arma::mat testSet;
  arma::Row<size_t> predictedLabels;
  size_t numClasses = 7;

  // Load data matrix.
      data::Load("data2.csv", trainingSet, true);
    data::Load("labels2.csv", labels, true);
    data::Load("labels2.csv", rawLabels, true);
// Load the model, if necessary.
  LinearSVMModel* model;
  model = new LinearSVMModel();


  // Now, do the training.

    data::NormalizeLabels(rawLabels, labels, model->mappings);

    numClasses=model->mappings.n_elem;
    model->svm.Lambda() = lambda;
    model->svm.Delta() = delta;
    model->svm.NumClasses() = numClasses;
    model->svm.FitIntercept() = intercept;

    if (optimizerType == "lbfgs")
    {
      ens::L_BFGS lbfgsOpt;
      lbfgsOpt.MaxIterations() = maxIterations;
      lbfgsOpt.MinGradientNorm() = tolerance;



      // This will train the model.
      model->svm.Train(trainingSet, labels, numClasses, lbfgsOpt);
    }
    else if (optimizerType == "psgd")
    {
      const double stepSize = CLI::GetParam<double>("step_size");
      const bool shuffle = !CLI::HasParam("shuffle");
      const size_t maxIt = epochs * trainingSet.n_cols;

      ens::ConstantStep decayPolicy(stepSize);

      #ifdef HAS_OPENMP
      size_t threads = omp_get_max_threads();
      #else
      size_t threads = 1;
      Log::Warn << "Using parallel SGD, but OpenMP support is "
                << "not available!" << endl;
      #endif

      ens::ParallelSGD<ens::ConstantStep> psgdOpt(maxIt, std::ceil(
        (float) trainingSet.n_cols / threads), tolerance, shuffle,
        decayPolicy);

      Log::Info << "Training model with ParallelSGD optimizer." << endl;

      // This will train the model.
      model->svm.Train(trainingSet, labels, numClasses, psgdOpt);
    }
  
    // Get the test dataset, and get predictions.
    data::Load("data2.csv", testSet, true);
    
    arma::Row<size_t> predictions;
    size_t trainingDimensionality;

    // Set the dimensionality according to fitintercept.
    if (intercept)
      trainingDimensionality = model->svm.Parameters().n_rows - 1;
    else
      trainingDimensionality = model->svm.Parameters().n_rows;

    // Checking the dimensionality of the test data.


    // Save class probabilities, if desired.
    
    
    
    model->svm.Classify(testSet, predictedLabels);
    data::RevertLabels(predictedLabels, model->mappings, predictions);

    // Calculate accuracy, if desired.


    // Save predictions, if desired.
    

  

  return 0;
}
