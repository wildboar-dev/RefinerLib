//--------------------------------------------------
// Unit Tests for class REngine
//
// @author: Wild Boar
//
// @date: 2022-04-07
//--------------------------------------------------

#include <gtest/gtest.h>

#include <RefinerLib/REngine.h>
using namespace NVL_App;

#include "../Helpers/PolyProblem.h"

//--------------------------------------------------
// Test Methods
//--------------------------------------------------

/**
 * @brief verify the error calculation
 */
TEST(REngine_Test, error_calculation)
{
	// Setup
	auto problem = PolyProblem(Vec<double, 5>(3,2,-7,1,6));
	problem.AddTrainPoint(1, 2);
	problem.AddTrainPoint(3, 4);
	Mat initialGuess = Mat_<double>::zeros(5, 1); 

	// Execute
	double error = problem.GetError(initialGuess, 0);

	// Confirm
	ASSERT_EQ(error, 225);
}

/**
 * @brief Confirm the generation of the Jacobian
 */
TEST(REngine_Test, confirm_jacobian)
{
	// Setup
	auto problem = new PolyProblem(Vec<double, 5>(3,2,-7,1,6));
	problem->AddTrainPoint(1, 2);
	auto refiner = REngine(problem);

	Mat initialGuess = Mat_<double>::zeros(5, 1);
	auto initialError = problem->GetError(initialGuess, 0); 

	// Execute
	Mat jacobian = refiner.GetJacobian(initialGuess, initialError, 0);
	auto output = (double *) jacobian.data;

	// Confirm
	ASSERT_EQ(jacobian.rows, 1);
	ASSERT_EQ(jacobian.cols, 5);
	ASSERT_NEAR(output[0], 30, 1e-3);
	ASSERT_NEAR(output[1], 30, 1e-3);
	ASSERT_NEAR(output[2], 120, 1e-3);
	ASSERT_NEAR(output[3], 60, 1e-3);
	ASSERT_NEAR(output[4], 30, 1e-3);
}

/**
 * @brief A single iterate test
 */
TEST(REngine_Test, test_iterate)
{
	// Setup
	auto problem = new PolyProblem(Vec<double, 5>(3,2,-7,1,6));

	problem->AddTrainPoint(0,0);
	problem->AddTrainPoint(-1,1);
	problem->AddTrainPoint(1,1);
	problem->AddTrainPoint(-2,2);
	problem->AddTrainPoint(2,-2);	
	
	auto refiner = REngine(problem);

	Mat param_0000 = Mat_<double>::zeros(5, 1);

	// Execute
	auto errors_0000 = refiner.GetErrors(param_0000);
	auto ave_error_0000 = refiner.GetAveError( errors_0000 );

	Mat param_0001 = refiner.Iterate(param_0000, errors_0000);

	auto errors_0001 = refiner.GetErrors(param_0001);
	auto ave_error_0001 = refiner.GetAveError( errors_0001 );

	Mat param_0002 = refiner.Iterate(param_0001, errors_0001);

	auto errors_0002 = refiner.GetErrors(param_0002);
	auto ave_error_0002 = refiner.GetAveError( errors_0002 );

	// Confirm
	ASSERT_LT(ave_error_0001[0], ave_error_0000[0]);
	ASSERT_LT(ave_error_0002[0], ave_error_0001[0]);
}

/**
 * @brief Perform minimization
 */
TEST(REngine_Test, test_full_minimize)
{
	// Setup
	auto problem = new PolyProblem(Vec<double, 5>(3,2,-7,1,6));

	problem->AddTrainPoint(0,0);
	problem->AddTrainPoint(10, 10);
	problem->AddTrainPoint(-200, 100);
	problem->AddTrainPoint(-244,3);
	problem->AddTrainPoint(700,-200);	
	
	auto refiner = REngine(problem);

	Mat params = Mat_<double>::zeros(5, 1);

	// Execute
	auto error = refiner.Minimize(params, 1000, 1e-8);
	auto pdata = (double *)params.data;

	// Confirm
	ASSERT_LT(error[0], 1e-3);
	ASSERT_NEAR(pdata[0], 3, 1e-3);
	ASSERT_NEAR(pdata[1], 2, 1e-3);
	ASSERT_NEAR(pdata[2], -7, 1e-3);
	ASSERT_NEAR(pdata[3], 1, 1e-3);
	ASSERT_NEAR(pdata[4], 6, 1e-3);
}
