package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLTensorFunction;
import com.github.neuralnetworks.util.Environment;

/**
 * Fill array with value
 */
public class OpenCLFill extends OpenCLTensorFunction
{
	private static final long serialVersionUID = 1L;

	protected float fillValue;

	public OpenCLFill(float fillValue)
	{
		super();
		this.fillValue = fillValue;
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

		int id = OpenCLCore.getInstance().Fill(deviceId, tensorRef.getId(), fillValue);
		return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
	}

	public float getFillValue()
	{
		return fillValue;
	}

	public void setFillValue(float fillValue)
	{
		this.fillValue = fillValue;
	}

	@Override
	public String kernelOptions(int order)
	{
		return " -D FiV" + order + "=" + fillValue;
	}
}
