#include "IceDogImage.h"

using namespace cv;
using namespace IceDogCVLib;

IceDogImage::IceDogImage()
{
}

IceDogImage::IceDogImage(ColorSpace colSpace,int imWidth, int imHeight, Pixel * data) :width(imWidth), height(imHeight),colorSpace(colSpace)
{
	imgData = std::shared_ptr<Pixel>(data, [](Pixel* ptr) {delete[] ptr; });
}

IceDogImage::~IceDogImage()
{
}

void IceDogImage::LoadFromUrl(std::string imgSource)
{
	colorSpace = ColorSpace::RGB;
	Mat cvImg= imread(imgSource, IMREAD_COLOR);
	LoadFromMat(cvImg);
}

void IceDogImage::LoadFromMat(cv::Mat mat)
{
	// set the width height
	width = mat.cols;
	height = mat.rows;

	// malloc the memory
	Pixel* targetData = new Pixel[width*height];
	// cast the Pixel ptr to uchar ptr
	auto targetDataUcharPtr= reinterpret_cast<uchar*>(targetData);
	// mem copy
	memcpy(targetDataUcharPtr, mat.data, sizeof(uchar) * 3 * width*height);
	// load the data to shared_ptr and assign the delete function
	imgData = std::shared_ptr<Pixel>(targetData, [](Pixel* pt) {delete[] pt; });
}

cv::Mat IceDogImage::StoreToMat()
{
	// alloc the mem space
	Mat imgMat( height,width, CV_8UC3, Scalar(0,0,0));
	// memcpy
	memcpy(imgMat.data, imgData.get(), sizeof(uchar) * 3 * width*height);
	return imgMat;
}

void IceDogImage::ConvertToHSV()
{
	if (colorSpace == ColorSpace::HSV) { return; }
	auto tempMat= StoreToMat();
	Mat hsvtemp(height, width, CV_8UC3, Scalar(0, 0, 0));
	// convert to hsv
	cvtColor(tempMat, hsvtemp, CV_BGR2HSV);
	// update the data
	LoadFromMat(hsvtemp);
	colorSpace = ColorSpace::HSV;
}

void IceDogCVLib::IceDogImage::ConvertToBGR()
{
	if (colorSpace == ColorSpace::RGB) { return; }
	auto tempMap = StoreToMat();
	Mat bgrtemp(height, width, CV_8UC3, Scalar(0, 0, 0));
	// convert to rgb
	cvtColor(tempMap, bgrtemp, CV_HSV2BGR);
	// update the data
	LoadFromMat(bgrtemp);
	colorSpace = ColorSpace::RGB;
}

std::vector<Pixel> IceDogCVLib::IceDogImage::FindClusterSeed(int numCluster)
{
	std::vector<Pixel> clusterCenter;
	// init randomer
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> intDist(0, width*height-1);
	clusterCenter.push_back(imgData.get()[intDist(gen)]);
	numCluster--;
	//
	while (numCluster!=0)
	{
		std::vector<double> distanceVec;
		for (int i=0;i<width*height;++i)
		{
			double distSum = 0;
			for (int j=0;j<clusterCenter.size();j++)
			{
				distSum+=imgData.get()[i].DistanceTo(clusterCenter[j]);
			}
			distanceVec.push_back(distSum);
		}
		//find the max distance
		int maxDisIndex = 0;
		for (int i=1;i<distanceVec.size();i++)
		{
			if (distanceVec[i] > distanceVec[maxDisIndex])
				maxDisIndex = i;
		}
		clusterCenter.push_back(imgData.get()[maxDisIndex]);
		numCluster--;
	}
	return clusterCenter;
}

IceDogImage IceDogImage::GenerateMostNearestMidValueFilterMap(int klSize)
{
	// define the color space in rgb
	if (colorSpace == ColorSpace::HSV)
	{
		ConvertToBGR();
		auto retImg = GenerateMostNearestMidValueFilterMap(klSize);
		ConvertToHSV();
		return retImg;
	}
	// create the target image
	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			Pixel maxOrMin;
			bool hasInit = false;
			float MidB = 0;
			float MidG = 0;
			float MidR = 0;
			for (int kh = 0; kh < klSize; ++kh)
			{
				for (int kw = 0; kw < klSize; ++kw)
				{
					int childW = w - (klSize - 1) / 2 + kw;
					int childH = h - (klSize - 1) / 2 + kh;
					if (childW < 0 || childW >= width || childH < 0 || childH >= height) { continue; }
					MidB += (float)(imgData.get()[childH*width + childW].BH) / ((float)(klSize*klSize));
					MidG += (float)(imgData.get()[childH*width + childW].GS) / ((float)(klSize*klSize));
					MidR += (float)(imgData.get()[childH*width + childW].RV) / ((float)(klSize*klSize));
				}
			}
			Pixel MidPixel(MidB, MidG, MidR);
			// kernel  proce
			for (int kh = 0; kh < klSize; ++kh)
			{
				for (int kw = 0; kw < klSize; ++kw)
				{
					int childW = w - (klSize - 1) / 2 + kw;
					int childH = h - (klSize - 1) / 2 + kh;
					if (childW < 0 || childW >= width || childH < 0 || childH >= height) { continue; }
					float klR = imgData.get()[childH*width + childW].RV;
					float klG = imgData.get()[childH*width + childW].GS;
					float klB = imgData.get()[childH*width + childW].BH;
					Pixel temp(klB, klG, klR);
					if (!hasInit)
					{
						maxOrMin = temp;
						hasInit = true;
					}
					else
					{
						if (temp.DistanceTo(MidPixel) < maxOrMin.DistanceTo(MidPixel))
							maxOrMin = temp;
					}
				}
			}
			newImgData[h*width + w] = maxOrMin;
		}
	}
	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

IceDogImage IceDogImage::ExpandCorrosionImage(Kernel procKernel,int flag)
{
	// define the color space in rgb
	if (colorSpace==ColorSpace::HSV)
	{
		ConvertToBGR();
		auto retImg= ExpandCorrosionImage(procKernel,flag);
		ConvertToHSV();
		return retImg;
	}
	// create the target image
	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			Pixel maxOrMin;
			bool hasInit = false;
			// kernel  proce
			for (int kh = 0; kh < procKernel.GetKernelSize(); ++kh)
			{
				for (int kw = 0; kw < procKernel.GetKernelSize(); ++kw)
				{
					int childW = w - (procKernel.GetKernelSize() - 1) / 2 + kw;
					int childH = h - (procKernel.GetKernelSize() - 1) / 2 + kh;
					if (childW < 0 || childW >= width || childH < 0 || childH >= height) { continue; }
					float klR = (float)(procKernel.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].RV);
					float klG = (float)(procKernel.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].GS);
					float klB = (float)(procKernel.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].BH);
					klR = klR < 0 ? 0 : klR;
					klG = klG < 0 ? 0 : klG;
					klB = klB < 0 ? 0 : klB;
					klR = klR > 255 ? 255 : klR;
					klG = klG > 255 ? 255 : klG;
					klB = klB > 255 ? 255 : klB;
					Pixel temp(klB, klG, klR);
					if (!hasInit)
					{
						maxOrMin = temp;
						hasInit = true;
					}
					else 
					{
						if (flag)
						{
							if (temp.GetGray() > maxOrMin.GetGray())
								maxOrMin = temp;
						}
						else
						{
							if (temp.GetGray() < maxOrMin.GetGray())
								maxOrMin = temp;
						}
					}
				}
			}
			newImgData[h*width + w] = maxOrMin;
		}
	}

	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

std::vector<IceDogCVLib::IceDogImage> IceDogImage::SplitImageUsingKeamsPP(int numCluster)
{
	std::vector<Pixel> clusterCenter = FindClusterSeed(numCluster);
	int* bClass=new int[width*height];
	// k-means proc
	while (true)
	{
		bool NoChange = true;
		// begin assign class
		for (int i=0;i<width*height;++i)
		{
			int minDistance = 0;
			for (int j = 1; j < clusterCenter.size(); j++)
			{
				if (clusterCenter[j].DistanceTo(imgData.get()[i]) < clusterCenter[minDistance].DistanceTo(imgData.get()[i]))
					minDistance = j;
			}
			if (minDistance != bClass[i])
			{
				bClass[i] = minDistance;
				NoChange = false;
			}
		}
		if (NoChange) { break; }
		// cal new center
		for (int i=0;i<clusterCenter.size();++i)
		{
			double bSum = 0;
			double gSum = 0;
			double rSum = 0;
			int count = 0;
			for (int j=0;j<width*height;++j)
				if (bClass[j] == i) { count++; }
			for (int j=0;j<width*height;++j)
			{
				if (bClass[j] == i)
				{
					bSum += ((double)imgData.get()[j].BH) / ((double)(count));
					gSum += ((double)imgData.get()[j].GS) / ((double)(count));
					rSum += ((double)imgData.get()[j].RV) / ((double)(count));
				}
			}
			clusterCenter[i].BH = bSum;
			clusterCenter[i].GS = gSum;
			clusterCenter[i].RV = rSum;
		}
	}
	// construct the result image
	std::vector<IceDogCVLib::IceDogImage> resultImg(numCluster);
	for (auto& img:resultImg)
	{
		img.width = width;
		img.height = height;
		img.colorSpace = colorSpace;
		Pixel* resultImgData = new Pixel[width*height];
		img.imgData = std::shared_ptr<Pixel>(resultImgData, [](Pixel* pt) {delete[] pt; });
	}
	// assign color for each class
	for (int i = 0; i < width*height; i++)
	{
		resultImg[bClass[i]].imgData.get()[i] = imgData.get()[i];
	}
	// re generate histogram
	for (auto& img:resultImg)
	{
		img.GenerateHistogram();
	}

	delete[] bClass;
	return resultImg;
}

void IceDogImage::SplitImageUsingOTSU(IceDogImage& frontImg, IceDogImage& bgImg)
{
	// only work in bgr color space
	if (colorSpace ==ColorSpace::HSV)
	{
		ConvertToBGR();
		SplitImageUsingOTSU(frontImg, bgImg);
		ConvertToHSV();
		return;
	}

	double histogram_gray[256];
	for (int i=0;i<256;i++)
		histogram_gray[i] = 0;
	double histSum = 0;
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			histSum += imgData.get()[h*width + w].GetGray();
			histogram_gray[(int)imgData.get()[h*width + w].GetGray()]++;
		}
	}
	// normilize the histogram
	for (int i = 0; i < 256; i++)
		histogram_gray[i] /= histSum;
	// use spT to split the image into two parts
	std::vector<double> fc;
	for (int i=1;i<256;i++)
	{
		double w0 = 0;
		double w1 = 0;
		double u0 = 0;
		double u1 = 0;
		for (int j=0;j<i;j++)
		{
			w0 += histogram_gray[j];
			u0 += histogram_gray[j] * j;
		}
		u0 /= w0;
		for (int j = i; j < 256; j++)
		{
			w1 += histogram_gray[j];
			u1 += histogram_gray[j] * j;
		}
		u1 /= w1;
		if (isnan(u0) || isnan(u1))
			fc.push_back(-1);
		else
			fc.push_back(w0*w1*(u0 - u1)*(u0 - u1));
	}
	int spPlace = 0;
	for (int i=1;i<fc.size();++i)
	{
		if (fc[i] > fc[spPlace]) { spPlace = i; }
	}
	// front image
	frontImg.width = width;
	frontImg.height = height;
	frontImg.colorSpace = colorSpace;
	Pixel* frontImgData = new Pixel[width*height];
	// back image
	bgImg.width = width;
	bgImg.height = height;
	bgImg.colorSpace = colorSpace;
	Pixel* bgImageData = new Pixel[width*height];

	for (int h=0;h<height;++h)
	{
		for (int w = 0; w < width; ++w)
		{
			if (imgData.get()[h*width + w].GetGray() < spPlace)
				frontImgData[h*width + w] = imgData.get()[h*width + w];
			else
				bgImageData[h*width + w] = imgData.get()[h*width + w];
		}
	}
	// assign front image data
	frontImg.imgData = std::shared_ptr<Pixel>(frontImgData, [](Pixel* pt) {delete[] pt; });
	// assign bg image data
	bgImg.imgData = std::shared_ptr<Pixel>(bgImageData, [](Pixel* pt) {delete[] pt; });

	frontImg.GenerateHistogram();
	bgImg.GenerateHistogram();
}

IceDogImage IceDogImage::Cut(int stX, int stY, int endX, int endY)
{
	// assert
	stX = stX < 0 ? 0 : stX;
	stY = stY < 0 ? 0 : stY;
	endX = endX >= width ? width - 1 : endX;
	endY = endY >= height ? height - 1 : endY;
	assert(stX < endX&&stY < endY);
	// create image head
	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = endY - stY + 1;
	targetImg.width = endX - stX + 1;
	// alloc the mem
	Pixel* newImgData = new Pixel[targetImg.height*targetImg.width];
	size_t dtPlace = 0;
	// update the pixel data
	for (int h=stY;h<=endY;++h)
	{
		for (int w=stX;w<=endX;++w)
		{
			newImgData[dtPlace] = imgData.get()[h*width + w];
			++dtPlace;
		}
	}
	// update the image
	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* pt) {delete[] pt; });
	return targetImg;
}

void IceDogImage::DrawCircle(int x, int y, float radius, Pixel color)
{
	/// only work in rgb space 
	if (colorSpace == ColorSpace::HSV)
	{
		ConvertToBGR();
		DrawCircle(x,y,radius,color);
		ConvertToHSV();
		return;
	}
	auto tempMat = StoreToMat();
	circle(tempMat, Point(x, y), radius, Scalar(color.BH, color.GS, color.RV));
	LoadFromMat(tempMat);
}

std::vector<std::tuple<float, float, float>> IceDogImage::FindCircleUsingHough()
{
	// use rgb color space
	if (colorSpace == ColorSpace::HSV)
	{
		ConvertToBGR();
		auto resultttt = FindCircleUsingHough();
		ConvertToHSV();
		return resultttt;
	}
	auto tempMat = StoreToMat();
	// convert to gray
	cvtColor(tempMat, tempMat, CV_BGR2GRAY);
	GaussianBlur(tempMat, tempMat, Size(9, 9), 2, 2);
	std::vector<std::tuple<float, float, float>> result;
	std::vector<Vec3f> test(20);
	HoughCircles(tempMat, test, CV_HOUGH_GRADIENT,2, tempMat.rows / 4, 200, 100);
	for (auto& cir:test)
	{
		result.push_back(std::tuple<float, float, float>(cir[0], cir[1], cir[2]));
	}
	tempMat.release();
	return result;
}

IceDogImage IceDogImage::GenerateDifferentEdgeMap(int threshold)
{
	// if is hsv just convert and do the job
	if (colorSpace == ColorSpace::HSV)
	{
		ConvertToBGR();
		auto resultttt = GenerateDifferentEdgeMap(threshold);
		ConvertToHSV();
		return resultttt;
	}
	// create the target image
	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			if (h + 1 >= height || w + 1 >= width) { newImgData[h*width + w] = Pixel(0, 0, 0); }
			else
			{
				float sumX, sumY, sumZ;
				sumX = std::abs((int)imgData.get()[(h + 1)*width + w].BH - (int)imgData.get()[h*width + w].BH) + std::abs((int)imgData.get()[h *width + w + 1].BH - (int)imgData.get()[h*width + w].BH);
				sumY = std::abs((int)imgData.get()[(h + 1)*width + w].GS - (int)imgData.get()[h*width + w].GS) + std::abs((int)imgData.get()[h *width + w + 1].GS - (int)imgData.get()[h*width + w].GS);
				sumZ = std::abs((int)imgData.get()[(h + 1)*width + w].RV - (int)imgData.get()[h*width + w].RV) + std::abs((int)imgData.get()[h *width + w + 1].RV - (int)imgData.get()[h*width + w].RV);
				sumX = sumX < threshold ? 0 : sumX;
				sumY = sumY < threshold ? 0 : sumY;
				sumZ = sumZ < threshold ? 0 : sumZ;
				sumX = sumX > 255 ? 255 : sumX;
				sumY = sumY > 255 ? 255 : sumY;
				sumZ = sumZ > 255 ? 255 : sumZ;
				newImgData[h*width + w].BH = sumX;
				newImgData[h*width + w].GS = sumY;
				newImgData[h*width + w].RV = sumZ;
			}
		}
	}

	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

IceDogImage IceDogImage::GeneratePrewittEdgeMap()
{
	// equal color space
	if (colorSpace == ColorSpace::HSV)
	{
		ConvertToBGR();
		auto resultttt = GeneratePrewittEdgeMap();
		ConvertToHSV();
		return resultttt;
	}
	// assign two kernel
	auto procKernelX = IceDogKernelGenerater::GeneratePrewittX3x3Kernel();
	auto procKernelY = IceDogKernelGenerater::GeneratePrewittY3x3Kernel();

	// create the target image
	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	for (int h = 1; h < height - 1; ++h)
	{
		for (int w = 1; w < width - 1; ++w)
		{
			float klXR = 0;
			float klXG = 0;
			float klXB = 0;

			float klYR = 0;
			float klYG = 0;
			float klYB = 0;
			// kernel  proce
			for (int kh = 0; kh < procKernelX.GetKernelSize(); ++kh)
			{
				for (int kw = 0; kw < procKernelX.GetKernelSize(); ++kw)
				{
					int childW = w - (procKernelX.GetKernelSize() - 1) / 2 + kw;
					int childH = h - (procKernelX.GetKernelSize() - 1) / 2 + kh;
					if (childW < 0 || childW >= width || childH < 0 || childH >= height) { continue; }
					klXR += (float)(procKernelX.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].RV);
					klXG += (float)(procKernelX.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].GS);
					klXB += (float)(procKernelX.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].BH);
				}
			}
			// kernel  proce
			for (int kh = 0; kh < procKernelY.GetKernelSize(); ++kh)
			{
				for (int kw = 0; kw < procKernelY.GetKernelSize(); ++kw)
				{
					int childW = w - (procKernelY.GetKernelSize() - 1) / 2 + kw;
					int childH = h - (procKernelY.GetKernelSize() - 1) / 2 + kh;
					if (childW < 0 || childW >= width || childH < 0 || childH >= height) { continue; }
					klYR += (float)(procKernelY.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].RV);
					klYG += (float)(procKernelY.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].GS);
					klYB += (float)(procKernelY.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].BH);
				}
			}
			klXB = std::sqrtf(klXB*klXB + klYB*klYB);
			klXG = std::sqrtf(klXG*klXG + klYG*klYG);
			klXR = std::sqrtf(klXR*klXR + klYR*klYR);
			newImgData[h*width + w] = Pixel(klXB > 255 ? 255 : klXB, klXG > 255 ? 255 : klXG, klXR > 255 ? 255 : klXR);
		}
	}

	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

IceDogImage IceDogImage::GenerateSobelEdgeMap()
{
	// equal color space
	if (colorSpace == ColorSpace::HSV)
	{
		ConvertToBGR();
		auto resultttt = GenerateSobelEdgeMap();
		ConvertToHSV();
		return resultttt;
	}
	// assign two kernel
	auto procKernelX = IceDogKernelGenerater::GenerateSobelX3x3Kernel();
	auto procKernelY = IceDogKernelGenerater::GenerateSobelY3x3Kernel();

	// create the target image
	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	for (int h = 1; h < height - 1; ++h)
	{
		for (int w = 1; w < width - 1; ++w)
		{
			float klXR = 0;
			float klXG = 0;
			float klXB = 0;

			float klYR = 0;
			float klYG = 0;
			float klYB = 0;
			// kernel  proce
			for (int kh = 0; kh < procKernelX.GetKernelSize(); ++kh)
			{
				for (int kw = 0; kw < procKernelX.GetKernelSize(); ++kw)
				{
					int childW = w - (procKernelX.GetKernelSize() - 1) / 2 + kw;
					int childH = h - (procKernelX.GetKernelSize() - 1) / 2 + kh;
					if (childW < 0 || childW >= width || childH < 0 || childH >= height) { continue; }
					klXR += (float)(procKernelX.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].RV);
					klXG += (float)(procKernelX.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].GS);
					klXB += (float)(procKernelX.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].BH);
				}
			}
			// kernel  proce
			for (int kh = 0; kh < procKernelY.GetKernelSize(); ++kh)
			{
				for (int kw = 0; kw < procKernelY.GetKernelSize(); ++kw)
				{
					int childW = w - (procKernelY.GetKernelSize() - 1) / 2 + kw;
					int childH = h - (procKernelY.GetKernelSize() - 1) / 2 + kh;
					if (childW < 0 || childW >= width || childH < 0 || childH >= height) { continue; }
					klYR += (float)(procKernelY.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].RV);
					klYG += (float)(procKernelY.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].GS);
					klYB += (float)(procKernelY.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].BH);
				}
			}
			klXB = std::sqrtf(klXB*klXB + klYB*klYB);
			klXG = std::sqrtf(klXG*klXG + klYG*klYG);
			klXR = std::sqrtf(klXR*klXR + klYR*klYR);
			newImgData[h*width + w] = Pixel(klXB>255?255:klXB,klXG>255?255:klXG,klXR>255?255:klXR);
		}
	}

	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

IceDogImage IceDogImage::GenerateLaplacianEdgeMap()
{
	if (colorSpace == ColorSpace::HSV)
	{
		ConvertToBGR();
		auto resultttt = GenerateLaplacianEdgeMap();
		ConvertToHSV();
		return resultttt;
	}
	auto procKernel = IceDogKernelGenerater::GenerateLaplacian3x3Kernel();

	// create the target image
	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	for (int h = 1; h < height-1; ++h)
	{
		for (int w = 1; w < width-1; ++w)
		{
			float klR = 0;
			float klG = 0;
			float klB = 0;
			// kernel  proce
			for (int kh = 0; kh < procKernel.GetKernelSize(); ++kh)
			{
				for (int kw = 0; kw < procKernel.GetKernelSize(); ++kw)
				{
					int childW = w - (procKernel.GetKernelSize() - 1) / 2 + kw;
					int childH = h - (procKernel.GetKernelSize() - 1) / 2 + kh;
					if (childW < 0 || childW >= width || childH < 0 || childH >= height) { continue; }
					klR += (float)(procKernel.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].RV);
					klG += (float)(procKernel.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].GS);
					klB += (float)(procKernel.GetKernelData(kh, kw))*(float)(imgData.get()[childH*width + childW].BH);
				}
			}
			newImgData[h*width + w] = Pixel(klB<0?0:klB, klG<0?0:klG, klR<0?0:klR);
		}
	}

	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

IceDogImage IceDogImage::operator-(IceDogImage& rightImg)
{
	// check the condition
	assert(colorSpace == rightImg.colorSpace);
	assert(width >= rightImg.width&&height >= rightImg.height);
	// create the image
	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	memcpy(newImgData, imgData.get(), sizeof(uchar) * 3 * width*height);
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			if (h < rightImg.height&&w < rightImg.width) { newImgData[h*width + w] = imgData.get()[h*width + w] - rightImg.imgData.get()[h*rightImg.width + w]; }
		}
	}
	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

IceDogImage IceDogImage::operator+(IceDogImage& rightImg)
{
	// check the condition
	assert(colorSpace == rightImg.colorSpace);
	assert(width >= rightImg.width&&height >= rightImg.height);
	// create the image
	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	memcpy(newImgData, imgData.get(), sizeof(uchar) * 3 * width*height);
	for (int h=0;h<height;++h)
	{
		for (int w=0;w<width;++w)
		{
			if (h < rightImg.height&&w < rightImg.width) { newImgData[h*width + w] = imgData.get()[h*width + w] + rightImg.imgData.get()[h*rightImg.width + w]; }
		}
	}
	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

IceDogImage IceDogImage::GenerateRobertsEdgeMap(int threshold)
{
	// if is hsv just convert and do the job
	if (colorSpace == ColorSpace::HSV)
	{
		ConvertToBGR();
		auto resultttt = GenerateRobertsEdgeMap(threshold);
		ConvertToHSV();
		return resultttt;
	}
	// create the target image
	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	for (int h=0;h<height;++h)
	{
		for (int w = 0; w < width; ++w)
		{
			if (h + 1 >= height || w + 1 >= width){ newImgData[h*width + w] = Pixel(0, 0, 0); }
			else
			{
				float sumX, sumY, sumZ;
				sumX = std::abs((int)imgData.get()[(h + 1)*width + w + 1].BH - (int)imgData.get()[h*width + w].BH) + std::abs((int)imgData.get()[h *width + w + 1].BH - (int)imgData.get()[(h + 1)*width + w].BH);
				sumY = std::abs((int)imgData.get()[(h + 1)*width + w + 1].GS - (int)imgData.get()[h*width + w].GS) + std::abs((int)imgData.get()[h *width + w + 1].GS - (int)imgData.get()[(h + 1)*width + w].GS);
				sumZ = std::abs((int)imgData.get()[(h + 1)*width + w + 1].RV - (int)imgData.get()[h*width + w].RV) + std::abs((int)imgData.get()[h *width + w + 1].RV - (int)imgData.get()[(h + 1)*width + w].RV);
				sumX = sumX < threshold ? 0 : sumX;
				sumY = sumY < threshold ? 0 : sumY;
				sumZ = sumZ < threshold ? 0 : sumZ;
				sumX = sumX > 255 ? 255 : sumX;
				sumY = sumY > 255 ? 255 : sumY;
				sumZ = sumZ > 255 ? 255 : sumZ;
				newImgData[h*width + w].BH = sumX;
				newImgData[h*width + w].GS = sumY;
				newImgData[h*width + w].RV = sumZ;
			}
		}
	}

	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

IceDogImage IceDogImage::GenerateGaussBlurFilterMap(int kernelSize, float yeta /*= 1.5*/)
{
	if (colorSpace == ColorSpace::HSV)
	{
		ConvertToBGR();
		auto resultttt = GenerateGaussBlurFilterMap(kernelSize, yeta);
		ConvertToHSV();
		return resultttt;
	}
	auto procKernel = IceDogKernelGenerater::GenerateGaussKernel(kernelSize,yeta);

	// create the target image
	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	Pixel* sourceImgData = imgData.get();
	auto process = [this,kernelSize,newImgData, sourceImgData](Kernel procKernel,int startIndexH, int endIndexH,int startIndexW,int endIndexW)
	{
		for (int h = startIndexH; h < endIndexH; ++h)
		{
			for (int w = startIndexW; w < endIndexW; ++w)
			{
				float klR = 0;
				float klG = 0;
				float klB = 0;
				// kernel  proce
				for (int kh = 0; kh < kernelSize; ++kh)
				{
					for (int kw = 0; kw < kernelSize; ++kw)
					{
						int childW = w - (kernelSize - 1) / 2 + kw;
						int childH = h - (kernelSize - 1) / 2 + kh;
						if (childW < 0 || childW >= width || childH < 0 || childH >= height) { continue; }
						klR += procKernel.GetKernelData(kh, kw)*(float)(imgData.get()[childH*width + childW].RV);
						klG += procKernel.GetKernelData(kh, kw)*(float)(imgData.get()[childH*width + childW].GS);
						klB += procKernel.GetKernelData(kh, kw)*(float)(imgData.get()[childH*width + childW].BH);
					}
				}
				newImgData[h*width + w] = Pixel(klB, klG, klR);
			}
		}
	};
	std::thread t1(process,procKernel,0,height/2,0,width/2);
	std::thread t2(process,procKernel,0,height/2,width/2,width);
	std::thread t3(process,procKernel,height/2,height,0,width/2);
	std::thread t4(process,procKernel,height/2,height,width/2,width);
	t1.join();
	t2.join();
	t3.join();
	t4.join();
	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

IceDogImage IceDogImage::GenerateOverMidValueFilterMap(int kernelSize, float T)
{
	// check kernel size
	assert(kernelSize > 0 && kernelSize % 2 != 0);
	// if not the correct colorSpace just change it
	if (colorSpace == ColorSpace::HSV)
	{
		ConvertToBGR();
		auto resultttt = GenerateOverMidValueFilterMap(kernelSize, T);
		ConvertToHSV();
		return resultttt;
	}

	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			std::vector<Pixel> orderList;
			// kernel  proce
			for (int kh = 0; kh < kernelSize; ++kh)
			{
				for (int kw = 0; kw < kernelSize; ++kw)
				{
					int childW = w - (kernelSize - 1) / 2 + kw;
					int childH = h - (kernelSize - 1) / 2 + kh;
					if (childW < 0 || childW >= width || childH < 0 || childH >= height) {continue; }
					orderList.push_back(imgData.get()[childH*width + childW]);
				}
			}
			// sort the list
			std::sort(orderList.begin(), orderList.end(), [](Pixel& p0,Pixel& p1)->bool 
			{
				return p0.Sum() < p1.Sum();
			});
			
			if (std::abs(imgData.get()[h*width + w].Sum() - orderList[orderList.size() / 2].Sum()) > T)
				newImgData[h*width + w] = orderList[orderList.size() / 2];
			else
				newImgData[h*width + w] = imgData.get()[h*width + w];
		}
	}
	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

IceDogImage IceDogImage::GenerateOverNeighborhoodAvgFilterMap(int kernelSize, float T)
{
	if (colorSpace == ColorSpace::HSV)
	{
		ConvertToBGR();
		auto resultttt= GenerateOverNeighborhoodAvgFilterMap(kernelSize,T);
		ConvertToHSV();
		return resultttt;
	}
	auto procKernel = IceDogKernelGenerater::GenerateMeanValueKernel(kernelSize);

	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	for (int h = 0; h < height; ++h)
	{
		for (int w=0;w<width;++w)
		{
			float klR = 0;
			float klG = 0;
			float klB = 0;
			// kernel  proce
			int skippedTimes = 0;
			for (int kh=0;kh<kernelSize;++kh)
			{
				for (int kw=0;kw<kernelSize;++kw)
				{
					int childW = w - (kernelSize - 1) / 2 + kw;
					int childH = h - (kernelSize - 1) / 2 + kh;
					if (childW < 0 || childW >= width || childH < 0 || childH >= height) { skippedTimes++; continue; }
					klR += procKernel.GetKernelData(kh, kw)*(float)(imgData.get()[childH*width + childW].RV);
					klG += procKernel.GetKernelData(kh, kw)*(float)(imgData.get()[childH*width + childW].GS);
					klB += procKernel.GetKernelData(kh, kw)*(float)(imgData.get()[childH*width + childW].BH);
				}
			}
			if (skippedTimes != 0) 
			{
				klR = (klR*kernelSize*kernelSize) / (float)(kernelSize*kernelSize - skippedTimes);
				klG = (klG*kernelSize*kernelSize) / (float)(kernelSize*kernelSize - skippedTimes);
				klB = (klB*kernelSize*kernelSize) / (float)(kernelSize*kernelSize - skippedTimes);
			}
			if (std::abs(imgData.get()[h*width + w].Sum()- (klR + klG + klB))>T)
				newImgData[h*width + w] = Pixel(klB, klG, klR);
			else
				newImgData[h*width + w] = imgData.get()[h*width + w];
		}
	}
	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

IceDogImage IceDogImage::GenerateGaussNoiseMap(float sigmoid)
{
	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	// init randomer
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> realDist(0.5, sigmoid);
	// get random noise
	for (size_t h = 0; h < height; ++h)
	{
		for (size_t w = 0; w < width; ++w)
		{
			float nois;
			nois = realDist(gen);
			if (nois > 1) { nois = 1; }
			if (nois < 0) { nois = 0; }
			newImgData[h*width + w].BH = imgData.get()[h*width + w].BH*nois;
			nois = realDist(gen);
			if (nois > 1) { nois = 1; }
			if (nois < 0) { nois = 0; }
			newImgData[h*width + w].GS = imgData.get()[h*width + w].GS*nois;
			nois = realDist(gen);
			if (nois > 1) { nois = 1; }
			if (nois < 0) { nois = 0; }
			newImgData[h*width + w].RV = imgData.get()[h*width + w].RV*nois;
		}
	}
	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

IceDogImage IceDogImage::GenerateRandomNoiseMap()
{
	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	// init randomer
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> realDist(0, 1);
	// get random noise
	for (size_t h = 0; h < height; ++h)
	{
		for (size_t w = 0; w < width; ++w)
		{
			newImgData[h*width + w].BH = imgData.get()[h*width + w].BH*realDist(gen);
			newImgData[h*width + w].GS = imgData.get()[h*width + w].GS*realDist(gen);
			newImgData[h*width + w].RV = imgData.get()[h*width + w].RV*realDist(gen);
		}
	}
	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

IceDogImage IceDogImage::GenerateSaltPepperNoiseMap(float probability)
{
	assert(probability > 0 && probability <= 1);
	// create a copy img
	IceDogImage targetImg;
	targetImg.colorSpace = colorSpace;
	targetImg.height = height;
	targetImg.width = width;
	Pixel* newImgData = new Pixel[width*height];
	// init randomer
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(0, (int)(1.0f / probability)-1);
	std::uniform_int_distribution<> binaryDist(0, 1);
	// get the salt pepper noise
	for (size_t h=0;h<height;++h)
	{
		for (size_t w=0;w<width;++w)
		{
			if (dist(gen) == 0)
			{
				if(binaryDist(gen)==0)
					newImgData[h*width + w] = Pixel(255, 255, 255);
			}
			else
				newImgData[h*width + w] = imgData.get()[h*width + w];
		}
	}
	targetImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* ptr) {delete[] ptr; });
	// update the generateHistogram
	targetImg.GenerateHistogram();
	return targetImg;
}

void IceDogImage::GenerateHistogram()
{
	//clean all histogram
	for (size_t i=0;i<256;i++)
	{
		histogram_BH[i] = 0;
		histogram_GS[i] = 0;
		histogram_RV[i] = 0;
	}
	// begin cal the histogram
	auto tempPtr = imgData.get();
	for (size_t h=0;h<height;++h)
	{
		for (size_t w=0;w<width;++w)
		{
			histogram_BH[tempPtr[h*width + w].BH]++;
			histogram_GS[tempPtr[h*width + w].GS]++;
			histogram_RV[tempPtr[h*width + w].RV]++;
		}
	}
}

Mat IceDogImage::DrawHisto2Mat(int imChanel)
{
	int* histData=nullptr;
	switch (imChanel)
	{
	case 0:
		histData = histogram_BH;
		break;
	case 1:
		histData = histogram_GS;
		break;
	case 2:
		histData = histogram_RV;
		break;
	}
	assert(histData != nullptr);
	assert(width != 0 && height != 0);
	Mat histo(300, 512, CV_8UC3, Scalar(0, 0, 0));
	// a function to draw line on the image
	auto drawLine = [&histo,this](int x,float hiH) 
	{
		// assuem that hiH has normilized;
		hiH =(1-hiH)*300;
		for (int row = 300-1; row >= (int)hiH; row--)
		{
			histo.data[row * 512 * 3 + x * 3 * 2 + 0] = 255;
			histo.data[row * 512 * 3 + x * 3 * 2 + 1] = 255;
			histo.data[row * 512 * 3 + x * 3 * 2 + 2] = 255;

			histo.data[(x * 2 + 1) * 3 + row * 512 * 3 + 0] = 255;
			histo.data[(x * 2 + 1) * 3 + row * 512 * 3 + 1] = 255;
			histo.data[(x * 2 + 1) * 3 + row * 512 * 3 + 2] = 255;
		}
	};
	
	int maxHi = histData[0];
	for (int i = 1; i < 256; i++)
	{
		if (histData[i] > maxHi) { maxHi = histData[i]; }
	}
	for (int i = 0; i < 256; ++i)
	{
		drawLine(i, (float)histData[i] / (float)maxHi);
	}
	GenerateHistogram();
	return histo;
}

IceDogImage IceDogImage::EqulizeHistogram()
{
	assert(colorSpace == ColorSpace::HSV);
	// get the hist data
	int* histData = histogram_RV;
	assert(histData != nullptr);
	assert(width != 0 && height != 0);
	// normilize the hist
	float histDataf[256];
	// cal the probility
	for (int i=0;i<256;i++)
	{
		histDataf[i] = (float)histData[i] / (float)(width*height);
	}
	// cal the prop sum hist
	float histDatafSum[256];
	for (auto& i : histDatafSum)
	{
		i = 0;
	}
	for (int i=0;i<256;i++)
	{
		for (int j = 0; j <= i; j++)
		{
			histDatafSum[i] += histDataf[j];
		}
	}
	// begin equlize the img
	Pixel* tempImPtr = imgData.get();
	Pixel* maxPixel;
	Pixel* minPixel;
	maxPixel = &tempImPtr[0];
	minPixel = &tempImPtr[0];
	// get the pixel that has max val
	if (colorSpace == ColorSpace::HSV)
	{
		for (int h = 0; h < height; ++h)
		{
			for (int w = 0; w < width; ++w)
			{
				if (tempImPtr[width*h + w].RV > maxPixel->RV) { maxPixel = (tempImPtr+width*h + w); }
				if (tempImPtr[width*h + w].RV < minPixel->RV) { minPixel = (tempImPtr+width*h + w); }
			}
		}
	}

	IceDogImage newImg;
	newImg.width = width;
	newImg.height = height;
	newImg.colorSpace = colorSpace;
	Pixel* newImgData = new Pixel[width*height];

	// equlize
	for (int h=0;h<height;h++)
	{
		for (int w = 0; w < width; w++)
		{
			Pixel targetPixel = tempImPtr[h*width + w];
			targetPixel.RV = histDatafSum[tempImPtr[h*width + w].RV] * (maxPixel->RV - minPixel->RV) + minPixel->RV;
			newImgData[h*width + w] = targetPixel;
		}
	}

	newImg.imgData = std::shared_ptr<Pixel>(newImgData, [](Pixel* pt) {delete[] pt; });
	newImg.GenerateHistogram();
	// after get the max value pixel
	return newImg;
}
