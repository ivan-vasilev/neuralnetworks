package com.github.neuralnetworks.calculation.operations.opencl;

import java.io.Serializable;

/**
 * Reference to opencl array
 */
public class OpenCLArrayReference implements Serializable
{
	private static final long serialVersionUID = 1L;

	private float[] array;
	private int id;
	private int deviceId;
	private boolean isModifiedOnDevice;
	private boolean isModifiedOnHost;

	public OpenCLArrayReference()
	{
		super();
		this.isModifiedOnHost = true;
	}

	public OpenCLArrayReference(float[] array, int id, int deviceId, boolean isModifiedOnDevice, boolean isModifiedOnHost)
	{
		super();
		this.array = array;
		this.id = id;
		this.deviceId = deviceId;
		this.isModifiedOnDevice = isModifiedOnDevice;
		this.isModifiedOnHost = isModifiedOnHost;
	}

	public float[] getArray()
	{
		return array;
	}

	public void setArray(float[] array)
	{
		this.array = array;
	}

	public int getId()
	{
		return id;
	}

	public void setId(int id)
	{
		this.id = id;
	}

	public int getDeviceId()
	{
		return deviceId;
	}

	public void setDeviceId(int deviceId)
	{
		this.deviceId = deviceId;
	}

	public boolean getIsModifiedOnDevice()
	{
		return isModifiedOnDevice;
	}

	public void setIsModifiedOnDevice(boolean isModifiedOnDevice)
	{
		this.isModifiedOnDevice = isModifiedOnDevice;
	}

	public boolean getIsModifiedOnHost()
	{
		return isModifiedOnHost;
	}

	public void setIsModifiedOnHost(boolean isModifiedOnHost)
	{
		this.isModifiedOnHost = isModifiedOnHost;
	}

	@Override
	public boolean equals(Object obj)
	{
		if (obj != null && obj instanceof OpenCLArrayReference)
		{
			OpenCLArrayReference o = (OpenCLArrayReference) obj;
			return id == o.id && deviceId == o.deviceId;
		}

		return false;
	}

	@Override
	public int hashCode()
	{
		return id * 31 + deviceId;
	}
}
