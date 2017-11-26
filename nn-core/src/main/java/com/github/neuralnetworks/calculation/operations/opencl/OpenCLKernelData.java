package com.github.neuralnetworks.calculation.operations.opencl;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;

import com.github.neuralnetworks.tensor.Tensor;

/**
 * Implementations have full knowledge over the required kernel
 */
public interface OpenCLKernelData extends Serializable
{
	public Tensor getInput();

	public Tensor getOutput();
	
	public String kernelOptions(int order);

	public OpenCLKernelReference createKernel();

	public void destroyKernel();

	public Integer getDeviceId();

	public void setDeviceId(Integer deviceId);

	public default Set<float[]> getModifiedArrays() {
		Set<float[]> result = new HashSet<>();
		result.add(getOutput().getElements());
		return result;
	}
}
