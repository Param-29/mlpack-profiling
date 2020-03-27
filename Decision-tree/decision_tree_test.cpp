#include<iostream>
#include <mlpack/prereqs.hpp>
#include<mlpack/methods/decision_tree/decision_tree.hpp>
//#include"decision_tree.hpp"
#include<omp.h>

using namespace std;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

int main(){
	arma::mat trainingSet;
  	arma::Row<size_t> labels;
	//REMOVE LAST COLUMN OF LABELS FROM DATA-SET AND ADD IT IN LABELS
  	data::Load("data2.csv", trainingSet, true);
  	data::Load("labels2.csv", labels, true);
  	cout<<"row";
  	//loading data
  	//while loading rows
  	//if column name present it gives error.
  	//parameters
  	const size_t numClasses = arma::max(arma::max(labels)) + 1;
  	const size_t minLeafSize = 5;
    const size_t maxDepth = 10000;
    const double minimumGainSplit =-1;

    cout<<trainingSet.n_cols<<endl<<endl;
    double t=omp_get_wtime();

    DecisionTree<> model = DecisionTree<>(trainingSet, labels,numClasses, minLeafSize, minimumGainSplit, maxDepth); //data-set not added.
  	//column names become 0 if it is a character for arma::mat and fatal error if arma::Row and names are present.
  	t=omp_get_wtime()-t;
    cout<<t<<endl;




	//testing on training data

    arma::Row<size_t> predictions;
    arma::mat probabilities;

    t=omp_get_wtime();
    model.Classify(trainingSet, predictions, probabilities);
    t=omp_get_wtime()-t;
    cout<<t<<endl;
  	size_t correct = 0;
      for (size_t i = 0; i < labels.n_cols; ++i)
        if (predictions[i] == labels[i])
          ++correct;

      // Print number of correct points.
      cout << double(correct) / double(trainingSet.n_cols) * 100 << "% "
          << "correct on test set (" << correct << " / " << trainingSet.n_cols
          << ")." << endl;
    /*      
  	//printing data
  	
  	for(size_t i=0;i<trainingSet.n_elem;i++){
  		cout<<trainingSet[i]<<endl;
  			
  	}

  	for(size_t i=0;i<labels.n_elem;i++)
  	cout<<labels[i]<<endl;
  	*/	
	return 0;
}
