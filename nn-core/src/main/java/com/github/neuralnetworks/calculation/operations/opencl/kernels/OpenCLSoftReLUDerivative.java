package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLTensorFunctionDerivative;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.util.Environment;

public class OpenCLSoftReLUDerivative extends OpenCLTensorFunctionDerivative
{

	private static final long serialVersionUID = 1L;
	
	public OpenCLSoftReLUDerivative()
	{
		super();
	}

	public OpenCLSoftReLUDerivative(Tensor activations)
	{
		super(activations);
	}

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

		OpenCLArrayReference activationsRef = rm.getArrayReference(activations, deviceId);
		if (activationsRef == null) {
			rm.addToDevice(deviceId, activations);
			activationsRef = rm.getArrayReference(activations, deviceId);
		}

		int id = OpenCLCore.getInstance().SoftReLUDerivative(deviceId, tensorRef.getId(), activationsRef.getId(), tensor.getSize(), activations.getStartIndex(), tensor.getStartIndex());
		return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
	}

	@Override
	public String kernelOptions(int order)
	{
		return " -D SUSI" + order + "=" + activations.getStartIndex() + " -D SUrS" + order + "=" + tensor.getStartIndex();
	}
}
