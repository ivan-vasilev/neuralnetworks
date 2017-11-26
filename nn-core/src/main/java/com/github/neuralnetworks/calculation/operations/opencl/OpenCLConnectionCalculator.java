package com.github.neuralnetworks.calculation.operations.opencl;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Util;

/**
 * base OpenCL connection calculator
 */
public abstract class OpenCLConnectionCalculator implements ConnectionCalculator, OpenCLKernelData
{
	private static final long serialVersionUID = 1L;

	protected transient Integer deviceId;
	protected Tensor input;
	protected Tensor output;
	protected int range;

	@Override
	public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		if (connections.size() != 1)
		{
			throw new RuntimeException("Illegal number of images");
		}

		input = TensorFactory.tensor(Util.getOppositeLayer(connections.get(0), targetLayer), connections.get(0), valuesProvider);
		output = TensorFactory.tensor(targetLayer, connections.get(0), valuesProvider);

		if (range == 0) {
			range = targetLayer.getUnitCount(connections);
		}

		OpenCLKernelsExecutor.getInstance().execute(this);
	}

	@Override
	public void destroyKernel()
	{ 
		input = output = null;
		range = 0;
	}

	@Override
	public Tensor getInput()
	{
		return input;
	}

	@Override
	public Tensor getOutput()
	{
		return output;
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
