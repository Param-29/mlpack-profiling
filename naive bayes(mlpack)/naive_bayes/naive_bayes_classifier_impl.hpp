/**
 * @file naive_bayes_classifier_impl.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Vahab Akbarzadeh (v.akbarzadeh@gmail.com)
 * @author Shihao Jing (shihao.jing810@gmail.com)
 *
 * A Naive Bayes Classifier which parametrically estimates the distribution of
 * the features.  This classifier makes its predictions based on the assumption
 * that the features have been sampled from a set of Gaussians with diagonal
 * covariance.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_CLASSIFIER_IMPL_HPP
#define MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_CLASSIFIER_IMPL_HPP

#include <mlpack/prereqs.hpp>

// In case it hasn't been included already.
#include "naive_bayes_classifier.hpp"

namespace mlpack {
namespace naive_bayes {

template<typename ModelMatType>
template<typename MatType>
NaiveBayesClassifier<ModelMatType>::NaiveBayesClassifier(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const bool incremental,
    const double epsilon) :
    trainingPoints(0), // Set when we call Train().
    epsilon(epsilon)
{
  static_assert(std::is_same<ElemType, typename MatType::elem_type>::value,
      "NaiveBayesClassifier: element type of given data must match the element "
      "type of the model!");

  // Perform training, after initializing the model to 0 (that is, if Train()
  // won't do that for us, which it won't if we're using the incremental
  // algorithm).
  if (incremental)
  {
    probabilities.zeros(numClasses);
    means.zeros(data.n_rows, numClasses);
    variances.zeros(data.n_rows, numClasses);
  }
  else
  {
    probabilities.set_size(numClasses);
    means.set_size(data.n_rows, numClasses);
    variances.set_size(data.n_rows, numClasses);
  }

  Train(data, labels, numClasses, incremental);
}

template<typename ModelMatType>
NaiveBayesClassifier<ModelMatType>::NaiveBayesClassifier(
    const size_t dimensionality,
    const size_t numClasses,
    const double epsilon) :
    trainingPoints(0),
    epsilon(epsilon)
{
  // Initialize model to 0.
  probabilities.zeros(numClasses);
  means.zeros(dimensionality, numClasses);
  variances.zeros(dimensionality, numClasses);
}

template<typename ModelMatType>
template<typename MatType>
void NaiveBayesClassifier<ModelMatType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const bool incremental)
{
  static_assert(std::is_same<ElemType, typename MatType::elem_type>::value,
      "NaiveBayesClassifier: element type of given data must match the element "
      "type of the model!");

  // Do we need to resize the model?
  if (probabilities.n_elem != numClasses)
  {
    // Perform training, after initializing the model to 0 (that is, if Train()
    // won't do that for us, which it won't if we're using the incremental
    // algorithm).
    if (incremental)
    {
      probabilities.zeros(numClasses);
      means.zeros(data.n_rows, numClasses);
      variances.zeros(data.n_rows, numClasses);
    }
    else
    {
      probabilities.set_size(numClasses);
      means.set_size(data.n_rows, numClasses);
      variances.set_size(data.n_rows, numClasses);
    }
  }

  // Calculate the class probabilities as well as the sample mean and variance
  // for each of the features with respect to each of the labels.
  if (incremental)
  {
    // Use incremental algorithm.
    // Fist, de-normalize probabilities.
    probabilities *= trainingPoints;

    for (size_t j = 0; j < data.n_cols; ++j)
    {
      const size_t label = labels[j];
      ++probabilities[label];

      arma::vec delta = data.col(j) - means.col(label);
      means.col(label) += delta / probabilities[label];
      variances.col(label) += delta % (data.col(j) - means.col(label));
      // compute intensive statement ( i.e try to parallelize)
    }

    for (size_t i = 0; i < probabilities.n_elem; ++i)
    {
      if (probabilities[i] > 2)
        variances.col(i) /= (probabilities[i] - 1);
    }
  }
  else
  {
    // Set all parameters to zero.
    probabilities.zeros();
    means.zeros();
    variances.zeros();

    // Don't use incremental algorithm.  This is a two-pass algorithm.  It is
    // possible to calculate the means and variances using a faster one-pass
    // algorithm but there are some precision and stability issues.  If this is
    // too slow, it's an option to use the faster algorithm by default and then
    // have this (and the incremental algorithm) be other options.

    // Calculate the means.
    for (size_t j = 0; j < data.n_cols; ++j)
    {
      const size_t label = labels[j];
      ++probabilities[label];
      means.col(label) += data.col(j);
    }

    // Normalize means.
    for (size_t i = 0; i < probabilities.n_elem; ++i)
      if (probabilities[i] != 0.0)
        means.col(i) /= probabilities[i];

    // Calculate variances.
    for (size_t j = 0; j < data.n_cols; ++j)
    {
      const size_t label = labels[j];
      variances.col(label) += square(data.col(j) - means.col(label));
      // compute intensive statement ( i.e try to parallelize)
    }

    // Normalize variances.
    for (size_t i = 0; i < probabilities.n_elem; ++i)
      if (probabilities[i] > 1)
        variances.col(i) /= (probabilities[i] - 1);
  }

  // Add epsilon to prevent log of zero.
  variances += epsilon;

  probabilities /= data.n_cols;
  trainingPoints += data.n_cols;
}

template<typename ModelMatType>
template<typename VecType>
void NaiveBayesClassifier<ModelMatType>::Train(const VecType& point,
                                               const size_t label)
{
  static_assert(std::is_same<ElemType, typename VecType::elem_type>::value,
      "NaiveBayesClassifier: element type of given data must match the element "
      "type of the model!");

  // We must use the incremental algorithm here.
  probabilities *= trainingPoints;
  probabilities[label]++;

  arma::vec delta = point - means.col(label);
  means.col(label) += delta / probabilities[label];
  if (probabilities[label] > 2)
    variances.col(label) *= (probabilities[label] - 2);
  variances.col(label) += (delta % (point - means.col(label)));
  if (probabilities[label] > 1)
    variances.col(label) /= probabilities[label] - 1;

  trainingPoints++;
  probabilities /= trainingPoints;
}

template<typename ModelMatType>
template<typename MatType>
void NaiveBayesClassifier<ModelMatType>::LogLikelihood(
    const MatType& data,
    ModelMatType& logLikelihoods) const
{
  static_assert(std::is_same<ElemType, typename MatType::elem_type>::value,
      "NaiveBayesClassifier: element type of given data must match the element "
      "type of the model!");

  logLikelihoods = arma::log(arma::repmat(probabilities, 1, data.n_cols));
  ModelMatType invVar = 1.0 / variances;

  // Calculate the joint log likelihood of point for each of the
  // means.n_cols.

  // Loop over every class.
  #pragma omp parallel for
  for (omp_size_t i = 0; i < means.n_cols; i++)
  {
    // This is an adaptation of gmm::phi() for the case where the covariance is
    // a diagonal matrix.
    ModelMatType diffs = data - arma::repmat(means.col(i), 1, data.n_cols);
    ModelMatType rhs = -0.5 * arma::diagmat(invVar.col(i)) * diffs;
    arma::Mat<ElemType> exponents = arma::sum(diffs % rhs, 0);
    logLikelihoods.row(i) += (data.n_rows / -2.0 * log(2 * M_PI) - 0.5 *
        arma::accu(arma::log(variances.col(i))) + exponents);  
    
  }
}

template<typename ModelMatType>
template<typename VecType>
size_t NaiveBayesClassifier<ModelMatType>::Classify(const VecType& point) const
{
  static_assert(std::is_same<ElemType, typename VecType::elem_type>::value,
      "NaiveBayesClassifier: element type of given data must match the element "
      "type of the model!");

  // Find the label(class) with max log likelihood.
  ModelMatType logLikelihoods;
  LogLikelihood(point, logLikelihoods);

  arma::uword maxIndex = 0;
  logLikelihoods.max(maxIndex);
  return maxIndex;
}

template<typename ModelMatType>
template<typename VecType, typename ProbabilitiesVecType>
void NaiveBayesClassifier<ModelMatType>::Classify(
    const VecType& point,
    size_t& prediction,
    ProbabilitiesVecType& probabilities) const
{
  static_assert(std::is_same<ElemType, typename VecType::elem_type>::value,
      "NaiveBayesClassifier: element type of given data must match the element "
      "type of the model!");
  static_assert(std::is_same<ElemType,
                             typename ProbabilitiesVecType::elem_type>::value,
      "NaiveBayesClassifier: element type of given data must match the element "
      "type of the model!");

  // log(Prob(Y|X)) = Log(Prob(X|Y)) + Log(Prob(Y)) - Log(Prob(X));
  // But LogLikelihood() gives us the unnormalized log likelihood which is
  // Log(Prob(X|Y)) + Log(Prob(Y)) so we need to subtract the normalization
  // term.
  ModelMatType logLikelihoods;
  LogLikelihood(point, logLikelihoods);

  // To prevent underflow in log of sum of exp of x operation (where x is a
  // small negative value), we use logsumexp(x - max(x)) + max(x).
  const double maxValue = arma::max(logLikelihoods);
  const double logProbX = log(arma::accu(exp(logLikelihoods - maxValue))) +
      maxValue;
  probabilities = exp(logLikelihoods - logProbX); // log(exp(value)) == value.

  arma::uword maxIndex = 0;
  logLikelihoods.max(maxIndex);
  prediction = (size_t) maxIndex;
}

template<typename ModelMatType>
template<typename MatType>
void NaiveBayesClassifier<ModelMatType>::Classify(
    const MatType& data,
    arma::Row<size_t>& predictions) const
{
  static_assert(std::is_same<ElemType, typename MatType::elem_type>::value,
      "NaiveBayesClassifier: element type of given data must match the element "
      "type of the model!");

  predictions.set_size(data.n_cols);

  ModelMatType logLikelihoods;
  LogLikelihood(data, logLikelihoods);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    arma::uword maxIndex = 0;
    logLikelihoods.unsafe_col(i).max(maxIndex);
    predictions[i] = maxIndex;
  }
}

template<typename ModelMatType>
template<typename MatType, typename ProbabilitiesMatType>
void NaiveBayesClassifier<ModelMatType>::Classify(
    const MatType& data,
    arma::Row<size_t>& predictions,
    ProbabilitiesMatType& predictionProbs) const
{
  static_assert(std::is_same<ElemType, typename MatType::elem_type>::value,
      "NaiveBayesClassifier: element type of given data must match the element "
      "type of the model!");
  static_assert(std::is_same<ElemType,
                             typename ProbabilitiesMatType::elem_type>::value,
      "NaiveBayesClassifier: element type of given data must match the element "
      "type of the model!");

  predictions.set_size(data.n_cols);

  ModelMatType logLikelihoods;
  LogLikelihood(data, logLikelihoods);

  predictionProbs.set_size(arma::size(logLikelihoods));
  #pragma omp parallel for
  for (omp_size_t j = 0; j < data.n_cols; ++j)
  {
    // The LogLikelihood() gives us the unnormalized log likelihood which is
    // Log(Prob(X|Y)) + Log(Prob(Y)), so we subtract the normalization term.
    // Besides, to prevent underflow in log of sum of exp of x operation (where
    // x is a small negative value), we use logsumexp(x - max(x)) + max(x).
    double maxValue = arma::max(logLikelihoods.col(j));
    double logProbX = log(arma::accu(exp(logLikelihoods.col(j) -
        maxValue))) + maxValue;
    predictionProbs.col(j) = arma::exp(logLikelihoods.col(j) - logProbX);
  }

  // Now calculate maximum probabilities for each point.
  #pragma omp parallel for 
  for (omp_size_t i = 0; i < data.n_cols; ++i)
  {
    arma::uword maxIndex = 0;
    logLikelihoods.unsafe_col(i).max(maxIndex);
    predictions[i] = maxIndex;
  }
}

template<typename ModelMatType>
template<typename Archive>
void NaiveBayesClassifier<ModelMatType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(means);
  ar & BOOST_SERIALIZATION_NVP(variances);
  ar & BOOST_SERIALIZATION_NVP(probabilities);
}

} // namespace naive_bayes
} // namespace mlpack

#endif
