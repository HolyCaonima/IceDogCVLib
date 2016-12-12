#include "IceDogImage.h"

using namespace cv;
using namespace std;
using namespace IceDogCVLib;

int main()
{
	IceDogImage img;
	img.LoadFromUrl("E:/color.png");
	IceDogImage fr, bg;
	auto imgs = img.GenerateGaussBlurFilterMap(13);
	imshow("mi", imgs.StoreToMat());
	cvWaitKey(0);
	return 0;
}
