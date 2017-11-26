package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLTensorFunctionDerivative;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.util.Environment;

public class OpenCLReLUDerivative extends OpenCLTensorFunctionDerivative
{

	private static final long serialVersionUID = 1L;
	
	public OpenCLReLUDerivative()
	{
		super();
	}

	public OpenCLReLUDerivative(Tensor activations)
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

		int id = OpenCLCore.getInstance().ReLUDerivative(deviceId, tensorRef.getId(), activationsRef.getId(), tensor.getSize(), activations.getStartIndex(), tensor.getStartIndex());
		return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
	}

	@Override
	public String kernelOptions(int order)
	{
		return " -D RDSI" + order + "=" + activations.getStartIndex() + " -D RDrS" + order + "=" + tensor.getStartIndex();
	}
}
