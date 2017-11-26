package com.github.neuralnetworks.calculation.operations.opencl;

import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.tensor.Tensor;

public class OpenCLArrayReferenceManager implements Serializable
{
	private static final long serialVersionUID = 1L;

	private static OpenCLArrayReferenceManager singleton = new OpenCLArrayReferenceManager();

	private transient Map<float[], Set<OpenCLArrayReference>> arrayReferences;

	private OpenCLArrayReferenceManager()
	{
		super();
		init();
	}

	private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException
	{
		stream.defaultReadObject();
		init();
	}

	private void init()
	{
		this.arrayReferences = new HashMap<>();
	}

	public static OpenCLArrayReferenceManager getInstance()
	{
		return singleton;
	}

	/**
	 * add array to the device
	 */
	public void addToDevice(int deviceId, float[] array, int offset)
	{
		OpenCLArrayReference ar = getArrayReference(array, deviceId);
		if (ar == null)
		{
			int id = OpenCLCore.getInstance().prepareFloatArray(deviceId, array, offset);
			addArrayReference(ar = new OpenCLArrayReference(array, id, deviceId, false, true));
		}
	}

	/**
	 * add tensor to the device
	 */
	public void addToDevice(int deviceId, Tensor... tensors)
	{
		for (Tensor t : tensors)
		{
			addToDevice(deviceId, t.getElements(), t.getStartOffset());
		}
	}

	/**
	 * push arrays to to the device (it has to be added first)
	 */
	public void pushToDevice(int deviceId, float[] array)
	{
		addToDevice(deviceId, array, 0);

		OpenCLArrayReference ar = getArrayReference(array, deviceId);
		if (ar != null && ar.getIsModifiedOnHost())
		{
			OpenCLCore.getInstance().updateFloatBuf(ar.getId(), array);
		}
	}

	/**
	 * push tensor to to the device (it has to be added first)
	 */
	public void pushToDevice(float[]... arrays)
	{
		for (float[] a : arrays)
		{
			Set<OpenCLArrayReference> references = getArrayReferences(a);
			if (references != null)
			{
				references.stream().filter(r -> r.getIsModifiedOnHost()).forEach(r -> OpenCLCore.getInstance().updateFloatBuf(r.getId(), a));
			}
		}
	}

	/**
	 * push tensor to to the device (it has to be added first)
	 * 
	 * @param t
	 */
	public void pushToDevice(int deviceId, Tensor... tensors)
	{
		for (Tensor t : tensors)
		{
			pushToDevice(deviceId, t.getElements());
		}
	}

	public void pushAllToDevice()
	{
		arrayReferences.entrySet().forEach(refs -> refs.getValue().forEach(ref -> pushToDevice(ref.getDeviceId(), ref.getArray())));
	}

	/**
	 * push arrays to the host (i.e. get the current value from the device)
	 */
	public synchronized void pushToHost(int deviceId, float[]... arrays)
	{
		for (float[] a : arrays)
		{
			OpenCLArrayReference ar = getArrayReference(a, deviceId);

			if (ar != null && ar.getIsModifiedOnDevice())
			{
				OpenCLCore.getInstance().getFloatBuf(ar.getId(), a);
				ar.setIsModifiedOnDevice(false);
			}
		}
	}

	/**
	 * push arrays to the host (i.e. get the current value from the device)
	 */
	public synchronized void pushToHost(float[]... arrays)
	{
		for (float[] a : arrays)
		{
			Set<OpenCLArrayReference> ar = getArrayReferences(a);

			if (ar == null)
			{
				throw new RuntimeException("Array not found on the device");
			}

			if (ar.size() != 1)
			{
				throw new RuntimeException("Array is linked to pushed to " + ar.size() + " instead of 1");
			}

			OpenCLArrayReference r = ar.iterator().next();
			if (r.getIsModifiedOnDevice())
			{
				OpenCLCore.getInstance().getFloatBuf(r.getId(), a);
				r.setIsModifiedOnDevice(false);
			}
		}
	}

	/**
	 * push tensors to the host (i.e. get the current value from the device)
	 */
	public synchronized void pushToHost(Tensor... tensors)
	{
		for (Tensor t : tensors)
		{
			pushToHost(t.getElements());
		}
	}

	/**
	 * push tensor arrays to the host (i.e. get the current value from the device)
	 */
	public synchronized void pushToHost(int deviceId, Tensor... tensors)
	{
		for (Tensor t : tensors)
		{
			pushToHost(deviceId, t.getElements());
		}
	}

	/**
	 * get the trained neural network information (tensors/arrays) from the open cl device
	 */
	public void pushAllToHost()
	{
		arrayReferences.entrySet().forEach(refs -> refs.getValue().forEach(ref -> pushToHost(ref.getDeviceId(), ref.getArray())));
	}

	public void pushAllToHost(int deviceId)
	{
		arrayReferences.entrySet().forEach(refs -> refs.getValue().stream().filter(ref -> ref.getDeviceId() == deviceId).forEach(ref -> pushToHost(ref.getDeviceId(), ref.getArray())));
	}

	public OpenCLArrayReference getArrayReference(float[] array, int deviceId)
	{
		OpenCLArrayReference result = null;
		Set<OpenCLArrayReference> references = arrayReferences.get(array);
		if (references != null)
		{
			result = references.stream().filter(r -> r.getDeviceId() == deviceId).findFirst().orElse(null);
		}

		return result;
	}

	public OpenCLArrayReference getArrayReference(Tensor t, int deviceId)
	{
		return getArrayReference(t.getElements(), deviceId);
	}

	public OpenCLArrayReference getArrayReference(int id, int deviceId)
	{
		for (Set<OpenCLArrayReference> refs : arrayReferences.values())
		{
			for (OpenCLArrayReference ref : refs)
			{
				if (ref.getId() == id && ref.getDeviceId() == deviceId)
				{
					return ref;
				}
			}
		}

		return null;
	}

	public Set<OpenCLArrayReference> getArrayReferences(float[] array)
	{
		return arrayReferences.get(array);
	}

	public Set<OpenCLArrayReference> getArrayReferences(Tensor t)
	{
		return arrayReferences.get(t.getElements());
	}

	public void addArrayReference(OpenCLArrayReference ref)
	{
		Set<OpenCLArrayReference> references = arrayReferences.get(ref.getArray());
		if (references == null)
		{
			arrayReferences.put(ref.getArray(), references = new HashSet<>());
		}

		references.add(ref);
	}

	public void clear()
	{
		arrayReferences.clear();
	}

	public void clearDevice(int deviceId)
	{
		arrayReferences.values().removeIf(s -> {
			s.removeIf(ref -> ref.getDeviceId() == deviceId);
			return s.isEmpty();
		});
	}
}
