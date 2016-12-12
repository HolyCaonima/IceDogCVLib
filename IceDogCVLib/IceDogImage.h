#pragma once

#include "IceDogKernelGenerater.h"

namespace IceDogCVLib
{
#define CHANEL_BH 0
#define CHANEL_GS 1
#define CHANEL_RV 2

	class IceDogImage
	{
	public:
		IceDogImage();
		IceDogImage(ColorSpace colSpace,int imWidth, int imHeight, Pixel* data);
		~IceDogImage();
	public:
		/* load image from url */
		void LoadFromUrl(std::string imgSource);
		/* load image from Mat */
		void LoadFromMat(cv::Mat mat);
		/* Store the image from imgData to mat */
		cv::Mat StoreToMat();
		/* convert the pixel data to hsv data */
		void ConvertToHSV();
		/* convert the pixel data to bgr data */
		void ConvertToBGR();
		/* generate the histogram */
		void GenerateHistogram();
		/* draw histogram to mat */
		cv::Mat DrawHisto2Mat(int imChanel);
		/* cut the image and return the sub image */
		IceDogImage Cut(int stX, int stY, int endX, int endY);
		/* equlize the image */
		IceDogImage EqulizeHistogram();
		/* generate salt pepper noise */
		IceDogImage GenerateSaltPepperNoiseMap(float probability);
		/* generate random noise */
		IceDogImage GenerateRandomNoiseMap();
		/* generate gauss noise */
		IceDogImage GenerateGaussNoiseMap(float sigmoid);
		/* generate over neighborhood averaging method filtered image */
		IceDogImage GenerateOverNeighborhoodAvgFilterMap(int kernelSize,float T= 140);
		/* generate over mid value method filtered image */
		IceDogImage GenerateOverMidValueFilterMap(int kernelSize, float T = 140);
		/* generate gauss blur filter map (add multi thread) */
		IceDogImage GenerateGaussBlurFilterMap(int kernelSize, float yeta = 1.5);
		/* generate Roberts Edge map */
		IceDogImage GenerateRobertsEdgeMap(int threshold=0);
		/* generate Laplacian edge map */
		IceDogImage GenerateLaplacianEdgeMap();
		/* generate Sobel edge map */
		IceDogImage GenerateSobelEdgeMap();
		/* generate Prewitt edge map */
		IceDogImage GeneratePrewittEdgeMap();
		/* generate different edge map */
		IceDogImage GenerateDifferentEdgeMap(int threshold = 0);
		/* Split image into front and bg two part using Otsu method */
		void SplitImageUsingOTSU(IceDogImage& frontImg, IceDogImage& bgImg);
		/* Split image into any class by using clustering k-means++ algorithm */
		std::vector<IceDogImage> SplitImageUsingKeamsPP(int numCluster);
		/* find class seed using keams++ algorithm */
		std::vector<Pixel> FindClusterSeed(int numCluster);
		/* find circle in image using hough method */
		std::vector<std::tuple<float, float, float>> FindCircleUsingHough();
		/* Expend or Corrosion image using a kernel */
		IceDogImage ExpandCorrosionImage(Kernel kl,int flag=1);
		/* most nearest mid value filter */
		IceDogImage GenerateMostNearestMidValueFilterMap(int klSize);
		/* draw circle on the image */
		void DrawCircle(int x, int y, float radius,Pixel color);
	private:
		/* to cv data position */
		int ToCVDataPosition(int x, int y) { return y*width * 3 + 3 * x; }

	public:
		/* add one image with another image */
		IceDogImage operator+(IceDogImage& rightImg);
		/* substract one image */
		IceDogImage operator-(IceDogImage& rightImg);

	public:
		/* Get current color space */
		ColorSpace GetColorSpace() { return colorSpace; }
		/* Get width */
		int GetWidth() { return width; }
		/* Get Height */
		int GetHeight() { return height; }

	private:
		// the image basic info
		int width = 0;
		int height = 0;
		// color space
		ColorSpace colorSpace;
		// data
		std::shared_ptr<Pixel> imgData;
		// histograms
		int histogram_BH[256];
		int histogram_GS[256];
		int histogram_RV[256];
	};
}

