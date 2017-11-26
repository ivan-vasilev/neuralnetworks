package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLTensorFunction;
import com.github.neuralnetworks.util.Environment;

public class OpenCLNoise extends OpenCLTensorFunction
{
	private static final long serialVersionUID = 1L;

	private final float corruptionLevel;
	private final float corruptedValue;

	public OpenCLNoise(float corruptionLevel, float corruptedValue)
	{
		super();
		this.corruptionLevel = corruptionLevel;
		this.corruptedValue = corruptedValue;
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

		OpenCLCore.getInstance().initPRNG(deviceId, tensor.getSize(), 0);

		int id = OpenCLCore.getInstance().Noise(deviceId, tensorRef.getId(), tensor.getSize(), tensor.getStartIndex(), corruptionLevel, corruptedValue);
		return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
	}

	@Override
	public String kernelOptions(int order)
	{
		return " -D NoaSI" + order + "=" + tensor.getStartIndex() + " -D NocL" + order + "=" + corruptionLevel + " -D NocV" + order + "=" + corruptedValue;
	}
}
