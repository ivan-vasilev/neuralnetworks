package com.github.neuralnetworks.calculation.operations.opencl;

import java.io.Serializable;
import java.util.Set;

import com.github.neuralnetworks.util.Environment;

/**
 * OpenCL method reference
 */
public class OpenCLKernelReference implements Serializable
{
	private static final long serialVersionUID = 1L;

	/**
	 * null for device-agnostic functions
	 */
	private Integer deviceId;
	private Integer id;

	/**
	 * References to the input and output arrays of the function
	 */
	private Set<float[]> modifiedArrays;

	public OpenCLKernelReference()
	{
		super();
	}

	public OpenCLKernelReference(Integer deviceId, int id, Set<float[]> modifiedArrays)
	{
		super();
		this.id = id;
		this.deviceId = deviceId;
		this.modifiedArrays = modifiedArrays;
	}

	public Integer getId()
	{
		return id;
	}

	public void setId(Integer id)
	{
		this.id = id;
	}

	public Integer getDeviceId()
	{
		return this.deviceId = this.deviceId != null ? this.deviceId : Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getPreferredDevice();
	}

	public void setDeviceId(Integer deviceId)
	{
		this.deviceId = deviceId;
	}

	public Set<float[]> getModifiedArrays()
	{
		return modifiedArrays;
	}

	public void setModifiedArrays(Set<float[]> modifiedArrays)
	{
		this.modifiedArrays = modifiedArrays;
	}
}
