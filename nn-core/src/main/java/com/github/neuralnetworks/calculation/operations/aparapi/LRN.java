package com.github.neuralnetworks.calculation.operations.aparapi;

import java.util.List;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Util;

/**
 * LRN feedforward phase
 * http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
 */
public class LRN implements ConnectionCalculator
{
	private static final long serialVersionUID = 1L;

	private LRNKernel kernel;

	// LRN constants
	private float k;
	private int n;
	private float a;
	private float b;

	public LRN(float k, int n, float a, float b)
	{
		super();
		this.k = k;
		this.n = n;
		this.a = a;
		this.b = b;
	}

	@Override
	public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		{
			Tensor input = TensorFactory.tensor(Util.getOppositeLayer(connections.get(0), targetLayer), connections.get(0), valuesProvider);
			Tensor output = TensorFactory.tensor(targetLayer, connections.get(0), valuesProvider);
			if (kernel == null)
			{
				kernel = new LRNKernel(input, output, k, n, a, b);
			}

			Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy()
					.execute(kernel, input.getDimensions()[1] * input.getDimensions()[2] * input.getDimensions()[3]);
		}
	}

	public float[] getCache()
	{
		return kernel != null ? kernel.cache : null;
	}

	public float getK()
	{
		return k;
	}

	public int getN()
	{
		return n;
	}

	public float getA()
	{
		return a;
	}

	public float getB()
	{
		return b;
	}

	public static class LRNKernel extends Kernel
	{
		protected float[] input;
		protected final int inputStartIndex;
		protected final int inputFeatureMaps;
		protected final int inputFeatureMapsLength;
		protected final int inputFeatureMapsDistance;
		protected final int inputMiniBatchDistance;
		protected final int miniBatchSize;

		protected Tensor outputTensor;
		protected float[] output;
		protected final int outputStartIndex;

		protected float[] cache;

		private final float k;
		private final int n;
		private final float a;
		private final float b;

		public LRNKernel(Tensor input, Tensor output, float k, int n, float a, float b)
		{
			this.input = input.getElements();
			this.inputStartIndex = input.getStartIndex();
			this.miniBatchSize = input.getDimensions()[0];
			this.inputFeatureMaps = input.getDimensions()[1];
			this.inputFeatureMapsLength = input.getDimensions()[2] * input.getDimensions()[3];
			this.inputMiniBatchDistance = input.getDimensionElementsDistance(0);
			this.inputFeatureMapsDistance = input.getDimensionElementsDistance(1);

			this.outputTensor = output;
			this.output = output.getElements();
			this.outputStartIndex = output.getStartIndex();

			this.cache = new float[input.getSize()];

			this.k = k;
			this.n = n;
			this.a = a;
			this.b = b;
		}

		@Override
		public void run()
		{
			int id = getGlobalId();
			int arrId = 0;
			int currentFM = id / inputFeatureMapsLength + 1;
			int startFM = (int) floor(max(0, currentFM - floor(n / 2) - 1));
			int fmCount = (int) (min(inputFeatureMaps, currentFM + floor(n / 2)) - startFM);
			int startId = inputStartIndex + startFM * inputFeatureMapsDistance + (id % inputFeatureMapsLength);
			int start = 0;
			float current = 0;
			float in = 0;

			for (int i = 0; i < miniBatchSize; i++)
			{
				current = 0;
				start = startId + inputMiniBatchDistance * i;
				for (int j = 0; j < fmCount; j++)
				{
					in = input[start + j * inputFeatureMapsDistance];
					current += in * in;
				}

				arrId = id + i * inputMiniBatchDistance;

				current = k + a * current;
				cache[arrId] = current;

				output[outputStartIndex + arrId] = input[inputStartIndex + arrId] / pow(current, b);
			}
		}

		public float[] getInput()
		{
			return input;
		}

		public int getInputStartIndex()
		{
			return inputStartIndex;
		}

		public int getInputFeatureMaps()
		{
			return inputFeatureMaps;
		}

		public int getInputFeatureMapsLength()
		{
			return inputFeatureMapsLength;
		}

		public int getInputFeatureMapsDistance()
		{
			return inputFeatureMapsDistance;
		}

		public int getMiniBatchSize()
		{
			return miniBatchSize;
		}

		public int getInputMiniBatchDistance()
		{
			return inputMiniBatchDistance;
		}

		public float[] getOutput()
		{
			return output;
		}

		public int getOutputStartIndex()
		{
			return outputStartIndex;
		}

		public float[] getCache()
		{
			return cache;
		}

		public Tensor getOutputTensor()
		{
			return outputTensor;
		}

		public float getK()
		{
			return k;
		}

		public int getN()
		{
			return n;
		}

		public float getA()
		{
			return a;
		}

		public float getB()
		{
			return b;
		}
	}
}
