package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLTensorFunction;
import com.github.neuralnetworks.util.Environment;

public class OpenCLReLU extends OpenCLTensorFunction
{
	private static final long serialVersionUID = 1L;

	@Override
	public OpenCLKernelReference createKernel()
	{
		this.deviceId = this.deviceId != null ? this.deviceId : Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getPreferredDevice();

		OpenCLArrayReferenceManager rm = OpenCLArrayReferenceManager.getInstance();

		OpenCLArrayReference tensorRef = rm.getArrayReference(tensor, deviceId);
		if (tensorRef == null) {
			rm.addToDevice(deviceId, tensor);
			tensorRef = rm.getArrayReference(tensor, deviceId);
		}

		int id = OpenCLCore.getInstance().ReLU(deviceId, tensorRef.getId(), tensor.getSize(), tensor.getStartIndex());
		return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
	}

	@Override
	public String kernelOptions(int order)
	{
		return " -D RLSI" + order + "=" + tensor.getStartIndex();
	}
}
