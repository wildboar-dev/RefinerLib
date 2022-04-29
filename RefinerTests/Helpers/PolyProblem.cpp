//--------------------------------------------------
// Implementation of class PolyProblem
//
// @author: Wild Boar
//
// @date: 2022-04-07
//--------------------------------------------------

#include "PolyProblem.h"
using namespace NVL_App;

//--------------------------------------------------
// Error Calculation
//--------------------------------------------------

/**
 * @brief Calculate the error associated with a given problem
 * @param params The parameters that we are optimizing
 * @param trainId The identifier of the problem that we are processing
 * @return double Returns a double
 */
double PolyProblem::GetError(Mat& params, int trainId)
{
	if (trainId < 0 || trainId >= _trainData.size()) throw runtime_error("ProblemId is out of range");
	auto x = _trainData[trainId][0];
	auto y = _trainData[trainId][1];
	auto expected = _trainData[trainId][2];
	auto actual = Evaluate(ExtractParams(params), x, y);

	auto difference = expected - actual;

	return difference * difference;
}

//--------------------------------------------------
// Problem Count
//--------------------------------------------------

/**
 * @brief Retrieve the number of problems
 * @return int Returns a int
 */
int PolyProblem::GetTrainingSize()
{
	return (int)_trainData.size();
}

//--------------------------------------------------
// Insert Problem
//--------------------------------------------------

/**
 * @brief Add a problem to the system
 * @param x The x component
 * @param y The y component
 */
void PolyProblem::AddTrainPoint(double x, double y) 
{
	auto solution = Evaluate(_expected, x, y);
	_trainData.push_back(Vec3d(x,y,solution));
}

//--------------------------------------------------
// Add the value that we are evaluating
//--------------------------------------------------

/**
 * @brief The value that we are evaluating
 * @param params The incomming parameters
 * @param x The x value
 * @param y The y value
 * @return double The evaluated value
 */
double PolyProblem::Evaluate(const Vec<double, 5>& params, double x, double y) 
{
	return params[0] * x * x + params[1] * x + params[2] * y * y + params[3] * y + params[4];
}

//--------------------------------------------------
// Add a helper for extracting parameters
//--------------------------------------------------

/**
 * @brief Add the logic for extracting parameters 
 * @param params The parameters that we are extracting
 * @return Vec<double, 5> The extracted value
 */
Vec<double, 5> PolyProblem::ExtractParams(Mat& params) 
{
	assert(params.rows == 5 && params.cols == 1);
	auto input = (double *) params.data;
	return Vec<double, 5>(input);
}