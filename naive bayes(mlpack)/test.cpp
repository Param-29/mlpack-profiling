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
#include<omp.h>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>
//#include "naive_bayes/naive_bayes_classifier.hpp"

using namespace mlpack;
using namespace mlpack::naive_bayes;
using namespace mlpack::util;
using namespace std;
using namespace arma;


#define MAXpoints 1000000
// Model loading/saving.

int main()
{
	const bool incrementalVariance = false;
	NaiveBayesClassifier<> model;
    mat trainingData ;
    Row<size_t> labels;
    Row<size_t> rawLabels;

    //Loading the data
    data::Load("data2.csv", trainingData, true);
    data::Load("labels2.csv", labels, true);
    data::Load("labels2.csv", rawLabels, true);
    
    labels=labels.cols(0,MAXpoints-1);
    rawLabels=labels.cols(0,MAXpoints-1);
    trainingData=trainingData.cols(0,MAXpoints-1);

    Col<size_t> mappings;  
    data::NormalizeLabels(rawLabels, labels, mappings);
	Row<size_t> predictions;
    mat probabilities;
    


    double Traintime=omp_get_wtime();
    model = NaiveBayesClassifier<>(trainingData, labels,
        mappings.n_elem, incrementalVariance);
    Traintime = omp_get_wtime() - Traintime;
    cout<<"\n"<<MAXpoints<<"\n"<<Traintime<<endl;

    double testTime=omp_get_wtime();
    model.Classify(trainingData, predictions, probabilities);
    testTime = omp_get_wtime() - testTime;
    cout<<"\n"<<MAXpoints<<"\n"<<testTime<<endl;
    
    size_t correct = 0;
      for (size_t i = 0; i < labels.n_cols; ++i)
        if (predictions[i] == labels[i])
          ++correct;

      // Print number of correct points.
      /*
      cout << double(correct) / double(trainingData.n_cols) * 100 << "% "
          << "correct on test set (" << correct << " / " << trainingData.n_cols
          << ")." << endl;

	*/

    return 0;
  
}

