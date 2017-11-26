package com.github.neuralnetworks.calculation.operations.opencl;

import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.util.Environment;

/**
 * Base class for opencl tensor functions
 */
public abstract class OpenCLTensorFunction implements TensorFunction, OpenCLKernelData
{
	private static final long serialVersionUID = 1L;

	protected transient Integer deviceId;

	protected Tensor tensor;

	@Override
	public Tensor getInput()
	{
		return tensor;
	}

	@Override
	public Tensor getOutput()
	{
		return tensor;
	}

	@Override
	public void value(Tensor inputOutput)
	{
		tensor = inputOutput;

		OpenCLKernelsExecutor.getInstance().execute(this);
	}

	@Override
	public void destroyKernel()
	{
		tensor = null;
	}

	@Override
	public Integer getDeviceId()
	{
		return this.deviceId = this.deviceId != null ? this.deviceId : Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getPreferredDevice();
	}

	@Override
	public void setDeviceId(Integer deviceId)
	{
		this.deviceId = deviceId;
	}
}
