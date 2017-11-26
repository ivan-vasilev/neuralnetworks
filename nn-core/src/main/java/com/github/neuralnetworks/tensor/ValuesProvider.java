package com.github.neuralnetworks.tensor;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.github.neuralnetworks.util.UniqueList;

import java.util.Set;

/**
 * Something similar to tensor factory
 */
public class ValuesProvider implements Serializable
{

	private static final long serialVersionUID = 1L;

	Set<Tensor> tensors;
	Map<Object, List<Tensor>> values;
	boolean useSharedMemory;

	public ValuesProvider(boolean useSharedMemory)
	{
		super();

		this.values = new HashMap<>();
		this.useSharedMemory = useSharedMemory;
		this.tensors = new LinkedHashSet<>();
	}

	public ValuesProvider(ValuesProvider sibling)
	{
		super();

		this.values = new HashMap<>();
		this.tensors = sibling.getTensors();
		this.useSharedMemory = sibling.useSharedMemory();
	}

	/**
	 * Get values for object based on provided dimensions
	 * @param key - the object to which the tenor belongs
	 * @param dimensions - the dimention of the tensor
	 * @return the first tensor found or null
	 */
	@SuppressWarnings("unchecked")
	public <T extends Tensor> T get(Object key, int... dimensions)
	{
		if (key != null && values.get(key) != null)
		{
			if (dimensions == null || dimensions.length == 0)
			{
				if (values.get(key).size() == 1)
				{
					return (T) values.get(key).iterator().next();
				} else if (values.get(key).size() > 1)
				{
					throw new IllegalArgumentException("No dimensions provided");
				} else
				{
					return null;
				}
			}

			return (T) values.get(key).stream().filter(t -> Arrays.equals(t.getDimensions(), dimensions)).findFirst().orElse(null);
		}

		return null;
	}

	/**
	 * @param t
	 * @return key for tensor
	 */
	public Object getKey(Tensor t)
	{
		Object result = null;
		for (Entry<Object, List<Tensor>> e : values.entrySet())
		{
			if (e.getValue().contains(t))
			{
				result = e.getKey();
				break;
			}
		}

		return result;
	}

	/**
	 * Add tensor t with dimensions
	 * 
	 * @param key
	 * @param dimensions
	 */
	public void add(Object key, int... dimensions)
	{
		add(key, false, dimensions);
	}

	/**
	 * Add tensor t with dimensions
	 * 
	 * @param key
	 * @param useSharedMemory
	 *          - will use the same underlying elements array, if size is the same
	 * @param dimensions
	 */
	public void add(Object key, boolean useSharedMemory, int... dimensions)
	{
		List<Tensor> set = values.get(key);
		if (set == null)
		{
			values.put(key, set = new UniqueList<Tensor>());
		}

		int size = Arrays.stream(dimensions).reduce(1, (a, b) -> a * b);
		float[] elements = null;
		int offset = 0;

		if (useSharedMemory)
		{
			Tensor sibling = set.stream().filter(t -> t.getSize() == size).findFirst().orElse(null);
			if (sibling != null)
			{
				elements = sibling.getElements();
				offset = sibling.getStartOffset();
			}
		}

		if (elements == null && useSharedMemory())
		{
			float[] oldElements = getElements();
			if (oldElements == null)
			{
				oldElements = new float[0];
			}

			offset = oldElements.length;
			elements = Arrays.copyOf(oldElements, offset + size);
			for (Tensor t : tensors)
				t.setElements(elements);
		}

		Tensor newTensor = null;

		if (elements != null)
		{
			tensors.add(newTensor = TensorFactory.tensor(elements, offset, dimensions));
		} else
		{
			tensors.add(newTensor = TensorFactory.tensor(dimensions));
		}

		set.add(newTensor);
	}
	
	/**
	 * Add tensor t for object
	 * 
	 * @param key
	 * @param tensor
	 */
	public void add(Object key, Tensor tensor)
	{
		List<Tensor> set = values.get(key);
		if (set == null)
		{
			values.put(key, set = new UniqueList<Tensor>());
		}

		set.add(tensor);
		tensors.add(tensor);
	}

	public Set<Tensor> getTensors()
	{
		return tensors;
	}

	public Map<Object, List<Tensor>> getValues()
	{
		return values;
	}

	public boolean useSharedMemory()
	{
		return useSharedMemory;
	}

	private float[] getElements()
	{
		float[] elements = null;
		for (Tensor t : tensors)
		{
			if (elements == null)
			{
				elements = t.getElements();
			}

			if (elements != t.getElements())
			{
				elements = null;
				break;
			}
		}

		return elements;
	}
}
