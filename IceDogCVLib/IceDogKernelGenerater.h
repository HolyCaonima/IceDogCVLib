#pragma once
#include "LibHead.h"

namespace IceDogCVLib
{
	class IceDogKernelGenerater
	{
	public:
		/* generate the mean value kernel */
		static Kernel GenerateMeanValueKernel(int kernelSize);
		/* generate the gauss kernel */
		static Kernel GenerateGaussKernel(int kernelSize, float yeta);
		/* generate laplacian 3x3 kernel */
		static Kernel GenerateLaplacian3x3Kernel();
		/* generate sobelX 3x3 kernel */
		static Kernel GenerateSobelX3x3Kernel();
		/* generate sobelY 3x3 kernel */
		static Kernel GenerateSobelY3x3Kernel();
		/* generate prewittX 3x3 kernel */
		static Kernel GeneratePrewittX3x3Kernel();
		/* generate prewittY 3x3 kernel */
		static Kernel GeneratePrewittY3x3Kernel();
		/* generate corrosion expand operation kernel */
		static Kernel GenerateECKernel(int kernelSize,float instMultipler = 1);
		/* generate corss expand corrosion operation kernel */	
		static Kernel GenerateCross3x3ECKernel(float instMultipler = 1);
	private:
		static inline float GaussValule2Dem(float h, float w, float yeta);
	};
}

