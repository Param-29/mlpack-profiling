/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file nbc_main.cpp
 *
 * This program runs the Simple Naive Bayes Classifier.
 *
 * This classifier does parametric naive bayes classification assuming that the
 * features are sampled from a Gaussian distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/data/normalize_labels.hpp>

#include <mlpack/methods/perceptron/perceptron.hpp>
using namespace mlpack;
using namespace mlpack::util;
using namespace std;
using namespace arma;
using namespace mlpack::perceptron;


// Model loading/saving.

int main()
{

    Perceptron<> model;
    mat trainingData ;
    Row<size_t> labels;
    Row<size_t> rawLabels;

    //Loading the data
    data::Load("data2.csv", trainingData, true);
    data::Load("labels2.csv", labels, true);
    data::Load("labels2.csv", rawLabels, true);

    Col<size_t> mappings;  
    data::NormalizeLabels(rawLabels, labels, mappings);
    
    const bool incrementalVariance = false;
    int maxIterations =100;
    int numClasses = 7;
    //cout<<labels.t();
    model = Perceptron<>(trainingData, labels, numClasses, maxIterations);
    model.Train(trainingData, labels, numClasses);

    mat testingData;
    data::Load("data2.csv", testingData, true);    


    Row<size_t> predictions;
    mat probabilities;
    
/*    model.Classify(testingData, predictions, probabilities);

    size_t correct = 0;
      for (size_t i = 0; i < labels.n_cols; ++i)
        if (predictions[i] == labels[i])
          ++correct;

      // Print number of correct points.
      cout << double(correct) / double(trainingData.n_cols) * 100 << "% "
          << "correct on test set (" << correct << " / " << trainingData.n_cols
          << ")." << endl;
*/    
    return 0;
  
}


