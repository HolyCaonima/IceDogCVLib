#include "IceDogKernelGenerater.h"

using namespace IceDogCVLib;

Kernel IceDogKernelGenerater::GenerateMeanValueKernel(int kernelSize)
{
	assert(kernelSize > 0&&kernelSize%2!=0);
	float* kernelData = new float[kernelSize*kernelSize];
	for (size_t i=0;i<kernelSize*kernelSize;++i)
	{
		kernelData[i] = 1.0 / (float)(kernelSize*kernelSize);
	}
	Kernel result(kernelSize, kernelData);
	return result;
}

Kernel IceDogKernelGenerater::GenerateGaussKernel(int kernelSize, float yeta)
{
	assert(kernelSize > 0 && kernelSize % 2 != 0);
	float* kernelData = new float[kernelSize*kernelSize];
	int place = 0;
	float klSum = 0;
	for (int h=-(kernelSize-1)/2;h<=(kernelSize-1)/2;h++)
	{
		for (int w=-(kernelSize-1)/2;w<=(kernelSize-1)/2;w++)
		{
			kernelData[place] = GaussValule2Dem(h, w, yeta);
			klSum += kernelData[place];
			++place;
		}
	}
	for (size_t i=0;i<kernelSize*kernelSize;i++)
	{
		kernelData[i] /= klSum;
	}
	return Kernel(kernelSize, kernelData);
}

float IceDogKernelGenerater::GaussValule2Dem(float h, float w, float yeta)
{
	return (1.0f / (2.0f*3.141592653f*yeta*yeta))*std::expf(-(h*h + w*w) / (2.0f*yeta*yeta));
}

Kernel IceDogKernelGenerater::GenerateLaplacian3x3Kernel()
{
	float* kernelData = new float[9]{0,-1,0,-1,4,-1,0,-1,0};
	Kernel result(3, kernelData);
	return result;
}

Kernel IceDogKernelGenerater::GenerateSobelX3x3Kernel()
{
	float* kernelData = new float[9]{ -1,0,1,-2,0,2,-1,0,1 };
	Kernel result(3, kernelData);
	return result;
}

Kernel IceDogKernelGenerater::GenerateSobelY3x3Kernel()
{
	float* kernelData = new float[9]{ 1,2,1,0,0,0,-1,-2,-1 };
	Kernel result(3, kernelData);
	return result;
}

Kernel IceDogKernelGenerater::GeneratePrewittX3x3Kernel()
{
	float* kernelData = new float[9]{ 1,1,1,0,0,0,-1,-1,-1 };
	Kernel result(3, kernelData);
	return result;
}

IceDogCVLib::Kernel IceDogCVLib::IceDogKernelGenerater::GeneratePrewittY3x3Kernel()
{
	float* kernelData = new float[9]{ -1,0,1,-1,0,1,-1,0,1 };
	Kernel result(3, kernelData);
	return result;
}

Kernel IceDogKernelGenerater::GenerateCross3x3ECKernel(float instMultipler /*= 1*/)
{
	float* kernelData = new float[9]{ 1*instMultipler,2*instMultipler,1*instMultipler,2*instMultipler ,2*instMultipler,2*instMultipler,1*instMultipler,2*instMultipler,1*instMultipler };
	Kernel result(3, kernelData);
	return result;
}

Kernel IceDogKernelGenerater::GenerateECKernel(int kernelSize,float instMultipler)
{
	float* kernelData = new float[kernelSize*kernelSize];
	for (int i=0;i<kernelSize*kernelSize;++i)
	{
		kernelData[i] =instMultipler;
	}
	Kernel result(kernelSize, kernelData);
	return result;
}
