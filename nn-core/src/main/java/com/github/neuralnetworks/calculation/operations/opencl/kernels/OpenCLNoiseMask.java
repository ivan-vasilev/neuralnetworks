package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import java.util.HashSet;
import java.util.Set;

import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLTensorFunction;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.util.Environment;

public class OpenCLNoiseMask extends OpenCLTensorFunction
{
	private static final long serialVersionUID = 1L;

	private Tensor mask;
	private final float corruptionLevel;
	private final float corruptedValue;

	public OpenCLNoiseMask(float corruptionLevel, float corruptedValue)
	{
		super();
		this.corruptionLevel = corruptionLevel;
		this.corruptedValue = corruptedValue;
	}

	@Override
	public void value(Tensor inputOutput)
	{
		if (mask == null || mask.getDimensions() != inputOutput.getDimensions()) 
		{
			mask = TensorFactory.tensor(inputOutput.getDimensions());
		}

		super.value(inputOutput);
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

		OpenCLArrayReference maskRef = rm.getArrayReference(mask, deviceId);
		if (maskRef == null)
		{
			rm.addToDevice(deviceId, mask);
			maskRef = rm.getArrayReference(mask, deviceId);
		}

		OpenCLCore.getInstance().initPRNG(deviceId, tensor.getSize(), 0);

		int id = OpenCLCore.getInstance().NoiseMask(deviceId, tensorRef.getId(), maskRef.getId(), tensor.getSize(), tensor.getStartIndex(), mask.getStartIndex(), corruptionLevel, corruptedValue);

		return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
	}

	@Override
	public Set<float[]> getModifiedArrays() {
		Set<float[]> result = new HashSet<>();
		result.add(getOutput().getElements());
		result.add(mask.getElements());

		return result;
	}

	@Override
	public void destroyKernel()
	{
		super.destroyKernel();
		mask = null;
	}

	public Tensor getMask()
	{
		return mask;
	}

	@Override
	public String kernelOptions(int order)
	{
		return " -D NmSI" + order + "=" + tensor.getStartIndex() + " -D NmmS" + order + "=" + mask.getStartIndex() + " -D NmCL" + order + "=" + corruptionLevel + " -D NmCV" + order + "=" + corruptedValue;
	}
}
