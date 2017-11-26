package com.github.neuralnetworks.tensor;

import java.io.Serializable;
import java.util.Arrays;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

import com.github.neuralnetworks.util.Util;

/**
 * N-dimensional tensor. For example 2-dim tensor is a matrix
 */
public class Tensor implements Serializable
{

	private static final long serialVersionUID = 1L;

	/**
	 * Start offset in case there is different dimentionality, i.e the values for his tenor in elements start
	 * at a different index, i.e. not 0. happen when multiple tensors use the same elements array
	 */
	protected int startOffset;

	/**
	 * tensor elements
	 */
	protected float[] elements;

	/**
	 * global dimension lengths for the tensor (based on the full elements array)
	 */
	protected int[] globalDimensions;

	/**
	 * local dimension size of the tensor based on the globalDimensions
	 * i.e. if the tensor is a subselections of a parent tensor
	 */
	protected int[] dimensions;

	/**
	 * size
	 */
	protected int size;

	/**
	 * sub-tensor position limits for each dimension
	 */
	protected int[][] globalDimensionsLimit;

	protected int[] dimMultiplicators;

	/**
	 * temporary array
	 */
	protected int[] dimTmp;

	/**
	 * creates a new tensor based on a parent tensor
	 * @param parent - the parent tensor
	 * @param dimensionsLimit -
	 * @param reduceChildDimensions
	 */
	public Tensor(Tensor parent, int[][] dimensionsLimit, boolean reduceChildDimensions)
	{
		this.startOffset = parent.startOffset;
		this.globalDimensions = parent.globalDimensions;
		this.elements = parent.elements;
		this.dimMultiplicators = parent.dimMultiplicators;
		this.globalDimensionsLimit = dimensionsLimit;
		this.dimTmp = new int[globalDimensions.length];

		this.dimensions = new int[reduceChildDimensions ? (int) IntStream.range(0, globalDimensions.length).filter(i -> dimensionsLimit[0][i] != dimensionsLimit[1][i]).count() : parent.dimensions.length];
		for (int i = 0, j = 0; i < globalDimensions.length; i++)
		{
			if (dimensionsLimit[0][i] != dimensionsLimit[1][i] || !reduceChildDimensions)
			{
				dimensions[j++] = dimensionsLimit[1][i] - dimensionsLimit[0][i] + 1;
			}
		}

		size = IntStream.range(0, dimensions.length).map(i -> dimensions[i]).reduce(1, (a, b) -> a * b);
	}

	public Tensor(int startOffset, float[] elements, int[] globalDimensions, int[][] globalDimensionsLimit)
	{
		super();

		if (globalDimensions == null || globalDimensions.length == 0)
		{
			throw new IllegalArgumentException("Please provide dimensions");
		}

		this.startOffset = startOffset;
		this.elements = elements;
		this.globalDimensions = globalDimensions;
		this.globalDimensionsLimit = globalDimensionsLimit;
		this.dimTmp = new int[globalDimensions.length];

		this.dimensions = new int[(int) IntStream.range(0, globalDimensions.length)
				.filter(i -> globalDimensionsLimit[0][i] != globalDimensionsLimit[1][i] || globalDimensionsLimit[1][i] - globalDimensionsLimit[0][i] + 1 == globalDimensions[i]).count()];
		for (int i = 0, j = 0; i < globalDimensions.length; i++)
		{
			if (globalDimensionsLimit[0][i] != globalDimensionsLimit[1][i] || globalDimensionsLimit[1][i] - globalDimensionsLimit[0][i] + 1 == globalDimensions[i])
			{
				dimensions[j++] = globalDimensionsLimit[1][i] - globalDimensionsLimit[0][i] + 1;
			}
		}

		this.dimMultiplicators = new int[dimensions.length];
		IntStream.range(0, dimMultiplicators.length).forEach(i -> {
			dimMultiplicators[i] = 1;
			Arrays.stream(globalDimensions).skip(i + 1).limit(globalDimensions.length).forEach(j -> dimMultiplicators[i] *= j);
		});

		size = IntStream.range(0, dimensions.length).map(i -> dimensions[i]).reduce(1, (a, b) -> a * b);
	}

	public float get(int... d)
	{
		return elements[getIndex(d)];
	}

	public void set(float value, int... d)
	{
		elements[getIndex(d)] = value;
	}

	/**
	 * @return Number of elements (may be different than elements.length)
	 */
	public int getSize()
	{
		return size;
	}

	/**
	 * @return start index (in the elements array) for this tensor
	 */
	public int getStartIndex()
	{
		Util.fillArray(dimTmp, 0);
		return getIndex(dimTmp);
	}

	/**
	 * @return end index (in the elements array) for this tensor
	 */
	public int getEndIndex()
	{
		IntStream.range(0, globalDimensions.length).forEach(i -> dimTmp[i] = globalDimensionsLimit[1][i] - globalDimensionsLimit[0][i]);
		return getIndex(dimTmp);
	}

	/**
	 * @param d
	 * @return the distance between two neighboring elements in this dimension in the elements array
	 */
	public int getDimensionElementsDistance(int d)
	{
		return dimMultiplicators[d];
	}

	public float[] getElements()
	{
		return elements;
	}

	public void setElements(float[] elements)
	{
		this.elements = elements;
	}

	public int[] getDimensions()
	{
		return dimensions;
	}

	public void setDimensions(int[] dimensions)
	{
		this.dimensions = dimensions;
	}

	public int getStartOffset()
	{
		return startOffset;
	}

	public void setStartOffset(int startOffset)
	{
		this.startOffset = startOffset;
	}

	/**
	 * @return iterator over the indexes of the elements array
	 */
	public TensorIterator iterator()
	{
		return new TensorIterator(this);
	}

	/**
	 * @return bordered iterator over the indexes of the elements array
	 */
	public TensorIterator iterator(int[][] limits)
	{
		return new TensorIterator(this, limits);
	}

	/**
	 * Iterate over all indexes
	 * 
	 * @param action
	 */
	public void forEach(IntConsumer action)
	{
		TensorIterator it = iterator();
		while (it.hasNext())
		{
			action.accept(it.next());
		}
	}

	protected int getIndex(int... d)
	{
		int id = 0;
		for (int i = 0, j = 0; i < globalDimensions.length; i++)
		{
			if (globalDimensionsLimit[0][i] != globalDimensionsLimit[1][i] || d.length - j >= globalDimensions.length - i)
			{
				if (d[j] + globalDimensionsLimit[0][i] > globalDimensionsLimit[1][i])
				{
					throw new IllegalArgumentException("Index out of range: " + i + " -> " + d[j] + "+" + globalDimensionsLimit[0][i] + " to " + globalDimensionsLimit[1][i]);
				}

				id += (d[j++] + globalDimensionsLimit[0][i]) * dimMultiplicators[i];
			} else
			{
				id += globalDimensionsLimit[0][i] * dimMultiplicators[i];
			}
		}

		return startOffset + id;
	}

	/**
	 * @param d
	 *          - dimension
	 * @return the index of this dimension within the global dimensions
	 */
	protected int getDimensionGlobalIndex(int d)
	{
		int result = d;

		if (IntStream.range(0, globalDimensions.length).filter(i -> globalDimensionsLimit[1][i] - globalDimensionsLimit[0][i] != globalDimensions[i]).findAny().isPresent())
		{
			for (int i = 0, dim = 0; i < globalDimensions.length; i++)
			{
				if (globalDimensionsLimit[0][i] != globalDimensionsLimit[1][i])
				{
					if (dim == d)
					{
						result = i;
						break;
					}

					dim++;
				}
			}
		}

		return result;
	}

	/**
	 * Iterate over the "real" indexes of the elements array
	 */
	public static class TensorIterator implements java.util.Iterator<Integer>
	{

		private Tensor tensor;
		private int[] currentPosition;
		private int[][] limits;

		public TensorIterator(Tensor tensor)
		{
			super();
			this.tensor = tensor;
			this.currentPosition = new int[tensor.dimensions.length];
			this.limits = new int[2][tensor.dimensions.length];
			IntStream.range(0, tensor.dimensions.length).forEach(i -> limits[1][i] = tensor.dimensions[i] - 1);
			currentPosition[currentPosition.length - 1] = -1;
		}

		public TensorIterator(Tensor tensor, int[][] limits)
		{
			super();
			this.tensor = tensor;
			this.currentPosition = new int[tensor.dimensions.length];
			this.limits = limits;
			IntStream.range(0, tensor.dimensions.length - 1).forEach(i -> currentPosition[i] = limits[0][i]);
			currentPosition[currentPosition.length - 1] = limits[0][currentPosition.length - 1] - 1;
		}

		@Override
		public boolean hasNext()
		{
			return IntStream.range(0, tensor.dimensions.length).anyMatch(i -> currentPosition[i] < limits[1][i]);
		}

		@Override
		public Integer next()
		{
			for (int d = tensor.dimensions.length - 1; d >= 0; d--)
			{
				if (currentPosition[d] != limits[1][d])
				{
					currentPosition[d]++;
					break;
				} else
				{
					currentPosition[d] = limits[0][d];
				}
			}

			return tensor.getIndex(currentPosition);
		}

		public int[] getCurrentPosition()
		{
			return currentPosition;
		}
	}
}
