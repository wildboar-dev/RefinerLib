//--------------------------------------------------
// A template for a refiner problem
//
// @author: Wild Boar
//
// @date: 2022-04-07
//--------------------------------------------------

#pragma once

#include <vector>
#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

#include <RefinerLib/RefinerProblem.h>

namespace NVL_App
{
	class PolyProblem : public RefinerProblem
	{
	private:
		Vec<double, 5> _expected;

		vector<Vec3d> _trainData;
	public:
		PolyProblem(const Vec<double, 5>& expected) : _expected(expected) {}

		void AddTrainPoint(double x, double y);

		virtual double GetError(Mat& params, int problemId) override;
		virtual int GetTrainingSize() override;
	private:
		double Evaluate(const Vec<double, 5>& params, double x, double y);
		Vec<double, 5> ExtractParams(Mat& params);
	};
}
