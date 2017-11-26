package com.github.neuralnetworks.tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.IntStream;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.RepeaterConnection;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.tensor.Tensor.TensorIterator;

public class TensorFactory
{

	@SuppressWarnings("unchecked")
	public static <T extends Tensor> T tensor(int... dimensions)
	{
		float[] elements = new float[IntStream.of(dimensions).reduce(1, (a, b) -> a * b)];
		int[][] dimensionsLimit = new int[2][dimensions.length];
		IntStream.range(0, dimensions.length).forEach(i -> dimensionsLimit[1][i] = dimensions[i] - 1);

		T result = null;
		if (dimensions.length == 2)
		{
			result = (T) new Matrix(0, elements, dimensions, dimensionsLimit);
		} else
		{
			result = (T) new Tensor(0, elements, dimensions, dimensionsLimit);
		}

		return result;
	}

	@SuppressWarnings("unchecked")
	public static <T extends Tensor> T tensor(float[] elements, int offset, int... dimensions)
	{
		int[][] dimensionsLimit = new int[2][dimensions.length];
		IntStream.range(0, dimensions.length).forEach(i -> dimensionsLimit[1][i] = dimensions[i] - 1);

		T result = null;
		if (dimensions.length == 2)
		{
			result = (T) new Matrix(offset, elements, dimensions, dimensionsLimit);
		} else
		{
			result = (T) new Tensor(offset, elements, dimensions, dimensionsLimit);
		}

		return result;
	}

	/**
	 * @param parent
	 * @param dimensionsLimit - limit over the parent dimensions
	 * @param reduceChildDimensions - reduce the child vector dimensions if they are limited to 1
	 * @return
	 */
	@SuppressWarnings("unchecked")
	public static <T extends Tensor> T tensor(Tensor parent, int[][] dimensionsLimit, boolean reduceChildDimensions)
	{
		T result = null;

		long dimensions = 0;
		if (reduceChildDimensions) {
			dimensions = IntStream.range(0, dimensionsLimit[0][0]).filter(i -> dimensionsLimit[0][i] != dimensionsLimit[1][i]).count();
		} else {
			dimensions = parent.getDimensions().length;
		}

		if (dimensions <= 2)
		{
			result = (T) new Matrix(parent, dimensionsLimit);
		} else
		{
			result = (T) new Tensor(parent, dimensionsLimit, reduceChildDimensions);
		}

		return result;
	}

	/**
	 * Create multiple combined tensors using shared elements array
	 * 
	 * @param dimensions
	 *          - dimensions for each tensor
	 * @return array of tensors
	 */
	public static Tensor[] tensor(int[]... dimensions)
	{
		Tensor[] result = new Tensor[dimensions.length];
		float[] elements = new float[Arrays.stream(dimensions).map(d -> {
			return IntStream.of(d).reduce(1, (a, b) -> a * b);
		}).reduce(0, (a, b) -> a + b)];

		for (int i = 0, offset = 0; i < dimensions.length; i++)
		{
			int[] d = dimensions[i];
			int[][] dimensionsLimit = new int[2][d.length];
			IntStream.range(0, d.length).forEach(j -> dimensionsLimit[1][j] = d[j] - 1);
			if (d.length == 2)
			{
				result[i] = new Matrix(offset, elements, d, dimensionsLimit);
			} else
			{
				result[i] = new Tensor(offset, elements, d, dimensionsLimit);
			}

			offset += IntStream.of(d).reduce(1, (a, b) -> a * b);
		}

		return result;
	}

	/**
	 * Simplified construction of matrix using values
	 * 
	 * @param elements
	 * @return Matrix
	 */
	public static Matrix matrix(float[][] elements)
	{
		Matrix result = tensor(elements[0].length, elements.length);
		IntStream.range(0, elements.length).forEach(i -> IntStream.range(0, elements[i].length).forEach(j -> {
			result.set(elements[i][j], i, j);
		}));

		return result;
	}

	public static Matrix matrix(float[] elements, int columns)
	{
		return tensor(elements, 0, elements.length / columns, columns);
	}

	public static void fill(Tensor t, float value)
	{
		TensorIterator it = t.iterator();
		while (it.hasNext())
		{
			t.getElements()[it.next()] = value;
		}
	}

	/**
	 * @param nn
	 * @param miniBatchSize
	 * @param useSharedMemory
	 * @return Tensor provider based on neural network
	 */
	public static ValuesProvider tensorProvider(NeuralNetwork nn, int miniBatchSize, boolean useSharedMemory)
	{
		ValuesProvider result = new ValuesProvider(useSharedMemory);

		createNNTensors(nn, result, miniBatchSize);

		return result;
	}

	/**
	 * @param miniBatchSize
	 * @param useSharedMemory
	 * @param nns
	 * @return Tensor provider based on multiple neural networks - common layers use shared tensors
	 */
	public static ValuesProvider tensorProvider(int miniBatchSize, boolean useSharedMemory, NeuralNetwork... nns)
	{
		ValuesProvider result = new ValuesProvider(useSharedMemory);

		for (NeuralNetwork nn : nns)
		{
			createNNTensors(nn, result, miniBatchSize);
		}

		return result;
	}

	/**
	 * @param sibling
	 * @param nn
	 * @return Tensor provider based on neural network
	 */
	public static ValuesProvider tensorProvider(ValuesProvider sibling, NeuralNetwork nn)
	{

		ValuesProvider result = new ValuesProvider(sibling);

		createNNTensors(nn, result, batchSize(sibling));

		return result;
	}

	public static void copy(Tensor src, Tensor dest)
	{
		if (!Arrays.equals(src.getDimensions(), dest.getDimensions()))
		{
			throw new IllegalArgumentException("Dimensions don't match");
		}

		TensorIterator srcIt = src.iterator();
		TensorIterator destIt = dest.iterator();
		while (srcIt.hasNext() && destIt.hasNext())
		{
			dest.getElements()[destIt.next()] = src.getElements()[srcIt.next()];
		}
	}
	
	/**
	 * Copy the contents of src to destination. The two providers must have equal keys and tensors
	 * 
	 * @param src
	 * @param dest
	 */
	public static void copy(ValuesProvider src, ValuesProvider dest)
	{
		for (Entry<Object, List<Tensor>> e : src.values.entrySet())
		{
			for (int i = 0; i < e.getValue().size(); i++)
			{
				copy(e.getValue().get(i), dest.values.get(e.getKey()).get(i));
			}
		}
	}

	public static Tensor duplicate(Tensor src, float[] elements)
	{
		int[][] gdl = new int[src.globalDimensionsLimit.length][src.globalDimensionsLimit[0].length];
		IntStream.range(0, gdl.length).forEach(i -> IntStream.range(0, gdl[i].length).forEach(j -> gdl[i][j] = src.globalDimensionsLimit[i][j]));

		if (elements == null)
		{
			elements = new float[src.elements.length];
		}

		Tensor result = new Tensor(src.startOffset, elements, Arrays.copyOf(src.globalDimensions, src.globalDimensions.length), gdl);

		return result;
	}

	/**
	 * Duplicate of the source values provider. The keys are the same as in "source" and the tensors are new
	 * 
	 * @param source
	 *          - source values provider
	 */
	public static ValuesProvider duplicate(ValuesProvider source)
	{
		ValuesProvider result = new ValuesProvider(source.useSharedMemory());
		Map<float[], float[]> elementsMap = new HashMap<>();
		for (Entry<Object, List<Tensor>> e : source.values.entrySet())
		{
			List<Tensor> r = new ArrayList<>();
			result.values.put(e.getKey(), r);

			for (Tensor st : e.getValue())
			{
				if (!elementsMap.containsKey(st.getElements()))
				{
					elementsMap.put(st.getElements(), new float[st.getElements().length]);
				}

				Tensor t = duplicate(st, elementsMap.get(st.getElements()));
				result.tensors.add(t);
				r.add(t);
			}
		}

		return result;
	}

	/**
	 * @return mini batch size for TensorProvider
	 */
	public static int batchSize(ValuesProvider tp)
	{
		Tensor t = tp.getTensors().iterator().next();
		return t.getDimensions()[0];
	}

	/**
	 * @return Tensor for connections. The connections must have a common layer and they must have the same dimensions.
	 */
	public static <T extends Tensor> T tensor(Layer targetLayer, Collection<Connections> connections, ValuesProvider tp)
	{
		return tp.get(targetLayer, getLayerDimensions(targetLayer, connections, batchSize(tp)));
	}

	/**
	 * @return Tensor for connections. The connections must have a common layer and they must have the same dimensions.
	 */
	public static <T extends Tensor> T tensor(Layer targetLayer, Connections c, ValuesProvider tp)
	{
		return tp.get(targetLayer, getLayerDimensions(targetLayer, Arrays.asList(new Connections[] { c }), batchSize(tp)));
	}

	/**
	 * tensor max element index
	 */
	public static int max(Tensor t)
	{
		int result = 0, i;
		float max = t.getElements()[t.getStartIndex()];
		TensorIterator it = t.iterator();
		while (it.hasNext())
		{
			i = it.next();
			if (t.getElements()[i] > max)
			{
				result = i;
				max = t.getElements()[i];
			}
		}

		return result;
	}

	/**
	 * tensor min element index
	 */
	public static int min(Tensor t)
	{
		int result = 0, i;
		float min = t.getElements()[t.getStartIndex()];
		TensorIterator it = t.iterator();
		while (it.hasNext())
		{
			i = it.next();
			if (t.getElements()[i] < min)
			{
				result = i;
				min = t.getElements()[i];
			}
		}
		
		return result;
	}

	/**
	 * average value of tensor elements
	 */
	public static float avg(Tensor t)
	{
		double result = 0;
		TensorIterator it = t.iterator();
		while (it.hasNext())
		{
			result += t.getElements()[it.next()];
		}

		return (float) (result / t.getSize());
	}
	
	/**
	 * @param targetLayer
	 * @param connections
	 * @return
	 */
	private static int[] getLayerDimensions(Layer targetLayer, Collection<Connections> connections, int miniBatchSize)
	{
		int[] result = null;
		boolean hasFullyConnected = false, hasSubsampling = false, hasConvolutional = false;
		for (Connections c : connections)
		{
			if (c instanceof FullyConnected)
			{
				hasFullyConnected = true;
			} else if (c instanceof Conv2DConnection)
			{
				hasConvolutional = true;
			} else if (c instanceof Subsampling2DConnection)
			{
				hasSubsampling = true;
			}
		}

		if (hasFullyConnected && (hasSubsampling || hasConvolutional))
		{
			throw new IllegalArgumentException("Cannot have fully connected and subsampling connections");
		}

		if (hasFullyConnected)
		{
			result = new int[] { miniBatchSize, targetLayer.getUnitCount(connections) };
		} else if (hasSubsampling)
		{
			Subsampling2DConnection c = (Subsampling2DConnection) connections.iterator().next();
			if (c.getOutputLayer() == targetLayer)
			{
				result = new int[] { miniBatchSize, c.getFilters(), c.getOutputFeatureMapRows(), c.getOutputFeatureMapColumns() };
			} else if (c.getInputLayer() == targetLayer)
			{
				result = new int[] { miniBatchSize, c.getFilters(), c.getInputFeatureMapRows(), c.getInputFeatureMapColumns() };
			}
		} else if (hasConvolutional)
		{
			Conv2DConnection c = (Conv2DConnection) connections.iterator().next();
			if (c.getOutputLayer() == targetLayer)
			{
				result = new int[] { miniBatchSize, c.getOutputFilters(), c.getOutputFeatureMapRows(), c.getOutputFeatureMapColumns() };
			} else if (c.getInputLayer() == targetLayer)
			{
				result = new int[] { miniBatchSize, c.getInputFilters(), c.getInputFeatureMapRows(), c.getInputFeatureMapColumns() };
			}
		}

		return result;
	}

	private static void createNNTensors(NeuralNetwork neuralNetwork, ValuesProvider vp, int miniBatchSize)
	{
		for (Connections c : neuralNetwork.getConnections())
		{
			if (c instanceof FullyConnected)
			{
				FullyConnected fc = (FullyConnected) c;

				int[] inputDim = new int[] { miniBatchSize, fc.getInputUnitCount() };
				if (vp.get(fc.getInputLayer(), inputDim) == null)
				{
					vp.add(fc.getInputLayer(), true, inputDim);
				}

				int[] outputDim = new int[] { miniBatchSize, fc.getOutputUnitCount() };
				if (vp.get(fc.getOutputLayer(), outputDim) == null)
				{
					vp.add(fc.getOutputLayer(), true, outputDim);
				}
			} else if (c instanceof Conv2DConnection)
			{
				Conv2DConnection cc = (Conv2DConnection) c;
				int[] inputDim = new int[] { miniBatchSize, cc.getInputFilters(), cc.getInputFeatureMapRows(), cc.getInputFeatureMapColumns() };
				if (vp.get(cc.getInputLayer(), inputDim) == null)
				{
					vp.add(cc.getInputLayer(), true, inputDim);
				}

				int[] outputDim = new int[] { miniBatchSize, cc.getOutputFilters(), cc.getOutputFeatureMapRows() + 2 * cc.getOutputRowPadding(),
						cc.getOutputFeatureMapColumns() + 2 * cc.getOutputColumnPadding() };
				if (cc.getOutputRowPadding() > 0 || cc.getOutputColumnPadding() > 0)
				{
					if (vp.get(cc.getOutputLayer(), outputDim) == null)
					{
						vp.add(cc.getOutputLayer(), true, outputDim);
						Tensor parent = vp.get(cc.getOutputLayer(), outputDim);

						int[] pd = parent.getDimensions();
						Tensor child = tensor(parent, new int[][] { { 0, 0, cc.getOutputRowPadding(), cc.getOutputColumnPadding() }, { pd[0] - 1, pd[1] - 1, pd[2] - cc.getOutputRowPadding() - 1, pd[3] - cc.getOutputColumnPadding() - 1 } }, false);
						vp.add(cc.getOutputLayer(), child);
					}
				} else if (vp.get(cc.getOutputLayer(), outputDim) == null)
				{
					vp.add(cc.getOutputLayer(), true, outputDim);
				}
			} else if (c instanceof Subsampling2DConnection)
			{
				Subsampling2DConnection cc = (Subsampling2DConnection) c;
				int[] inputDim = new int[] { miniBatchSize, cc.getFilters(), cc.getInputFeatureMapRows(), cc.getInputFeatureMapColumns() };
				if (vp.get(cc.getInputLayer(), inputDim) == null)
				{
					vp.add(cc.getInputLayer(), true, inputDim);
				}

				int[] outputDim = new int[] { miniBatchSize, cc.getFilters(), cc.getOutputFeatureMapRows() + 2 * cc.getOutputRowPadding(), cc.getOutputFeatureMapColumns() + 2 * cc.getOutputColumnPadding() };
				if (cc.getOutputRowPadding() > 0 || cc.getOutputColumnPadding() > 0)
				{
					if (vp.get(cc.getOutputLayer(), outputDim) == null)
					{
						vp.add(cc.getOutputLayer(), true, outputDim);
						Tensor parent = vp.get(cc.getOutputLayer(), outputDim);

						int[] pd = parent.getDimensions();
						Tensor child = tensor(parent, new int[][] { { 0, 0, cc.getOutputRowPadding(), cc.getOutputColumnPadding() }, { pd[0] - 1, pd[1] - 1, pd[2] - cc.getOutputRowPadding() - 1, pd[3] - cc.getOutputColumnPadding() - 1 } }, false);
						vp.add(cc.getOutputLayer(), child);
					}
				} else if (vp.get(cc.getOutputLayer(), outputDim) == null)
				{
					vp.add(cc.getOutputLayer(), true, outputDim);
				}
			} else if (c instanceof RepeaterConnection)
			{
				if (vp.get(c.getInputLayer()) != null)
				{
					int[] sourceDim = vp.get(c.getInputLayer()).getDimensions();
					if (vp.get(c.getOutputLayer(), sourceDim) == null)
					{
						vp.add(c.getOutputLayer(), true, sourceDim);
					}
				}

				if (vp.get(c.getOutputLayer()) != null)
				{
					int[] sourceDim = vp.get(c.getOutputLayer()).getDimensions();
					if (vp.get(c.getInputLayer(), sourceDim) == null)
					{
						vp.add(c.getInputLayer(), true, sourceDim);
					}
				}
			}
		}
	}
}
