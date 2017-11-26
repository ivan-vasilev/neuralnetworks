package com.github.neuralnetworks.calculation.operations.opencl;

import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Reference manager for kernel references
 */
public class OpenCLKernelReferenceManager implements Serializable
{
	private static final long serialVersionUID = 1L;

	private static OpenCLKernelReferenceManager singleton = new OpenCLKernelReferenceManager();

	private transient Map<Object, OpenCLKernelReference> kernelReferences;

	private OpenCLKernelReferenceManager()
	{
		super();
		this.kernelReferences = new HashMap<>();
	}

	private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException
	{
		stream.defaultReadObject();
		init();
	}

	private void init()
	{
		this.kernelReferences = new HashMap<>();
	}

	public static OpenCLKernelReferenceManager getInstance()
	{
		return singleton;
	}

	public Map<Object, OpenCLKernelReference> getKernelReferences()
	{
		return kernelReferences;
	}

	public OpenCLKernelReference get(Object key)
	{
		return kernelReferences.get(key);
	}

	public OpenCLKernelReference add(Object key, int deviceId, int id, int... arrayIds)
	{
		OpenCLKernelReference reference = null;

		if (id > 0)
		{
			Set<float[]> modifiedArrays = null;
			if (arrayIds != null && arrayIds.length > 0)
			{
				modifiedArrays = new HashSet<>();
				for (Integer i : arrayIds)
				{
					modifiedArrays.add(OpenCLArrayReferenceManager.getInstance().getArrayReference(i, deviceId).getArray());
				}
			}

			reference = new OpenCLKernelReference(deviceId, id, modifiedArrays);

			OpenCLKernelReferenceManager.getInstance().put(key, reference);
		}

		return reference;
	}

	public void put(Object key, OpenCLKernelReference reference)
	{
		kernelReferences.put(key, reference);
	}

	public void clear()
	{
		kernelReferences.clear();
	}

	public void clearDevice(int deviceId)
	{
		kernelReferences.entrySet().stream().filter(e -> e.getValue().getDeviceId() != null && e.getValue().getDeviceId() == deviceId).forEach(e -> {
			e.getValue().setDeviceId(null);
			e.getValue().setId(null);
		});

		kernelReferences.entrySet().removeIf(e -> e.getValue().getDeviceId() != null && e.getValue().getDeviceId() == deviceId);
	}
}
