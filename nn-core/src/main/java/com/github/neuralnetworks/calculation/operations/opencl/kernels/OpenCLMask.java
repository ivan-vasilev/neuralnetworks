package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLTensorFunction;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.util.Environment;

public class OpenCLMask extends OpenCLTensorFunction
{
	private static final long serialVersionUID = 1L;

	private OpenCLNoiseMask noiseMask;

	public OpenCLMask(OpenCLNoiseMask noiseMask)
	{
		super();
		this.noiseMask = noiseMask;
	}

	@Override
	public OpenCLKernelReference createKernel()
	{
		this.deviceId = this.deviceId != null ? this.deviceId : Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getPreferredDevice();

		OpenCLArrayReferenceManager rm = OpenCLArrayReferenceManager.getInstance();

		OpenCLArrayReference tensorRef = rm.getArrayReference(tensor, deviceId);
		if (tensorRef == null)
		{
			rm.addToDevice(deviceId, tensor);
			tensorRef = rm.getArrayReference(tensor, deviceId);
		}

		Tensor mask = noiseMask.getMask();
		OpenCLArrayReference maskRef = rm.getArrayReference(mask, deviceId);
		if (maskRef == null)
		{
			rm.addToDevice(deviceId, mask);
			maskRef = rm.getArrayReference(mask, deviceId);
		}

		int id = OpenCLCore.getInstance().Mask(deviceId, tensorRef.getId(), maskRef.getId(), tensor.getSize(), tensor.getStartIndex(), mask.getStartIndex());

		return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
	}

	@Override
	public String kernelOptions(int order)
	{
		return " -D MKSI" + order + "=" + tensor.getStartIndex() + " -D MKmS" + order + "=" + noiseMask.getMask().getStartIndex();
	}
}
