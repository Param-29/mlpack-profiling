#include<iostream>
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/rnn.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::ann;

int main(){
arma::mat dataset;
data::Load("thyroid_train.csv", dataset, true);
// Split the labels from the training set.
arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 4,
    dataset.n_cols - 1);
// Split the data from the training set.
arma::mat trainLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
    dataset.n_rows - 1, dataset.n_cols - 1);
// Initialize the network.
FFN<> model;
model.Add<Linear<> >(trainData.n_rows, 8);
model.Add<SigmoidLayer<> >();
model.Add<Linear<> >(8, 3);
model.Add<LogSoftMax<> >();
// Train the model.
model.Train(trainData, trainLabelsTemp);
// Use the Predict method to get the assignments.
arma::mat assignments;
model.Predict(trainData, assignments);
return 0;
}

/*
CPU UTILIZATION 0.98

*/