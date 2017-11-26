package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import java.util.List;
import java.util.Map;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.RepeaterConnection;
import com.github.neuralnetworks.calculation.operations.aparapi.LRN;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculatorImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Util;

/**
 * LRN backprop phase
 * http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
 */
public class BackPropagationLRN extends BackPropagationConnectionCalculatorImpl
{
	private static final long serialVersionUID = 1L;

	private LRN lrn;

	public BackPropagationLRN(Properties properties, LRN lrn)
	{
		super(properties);
		this.lrn = lrn;
	}

	@Override
	protected void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider,
			ValuesProvider activations, Layer targetLayer)
	{
		RepeaterConnection con = null;
		for (Connections c : inputConnections)
		{
			if (c instanceof RepeaterConnection)
			{
				con = (RepeaterConnection) c;
				break;
			}
		}

		if (con != null)
		{
			connectionCalculators.put(con, new BPLRNKernel(valuesProvider, activations, con, targetLayer, lrn.getCache(), lrn.getN(), lrn.getA(), lrn.getB()));
		}
	}

	public static class BPLRNKernel extends Kernel implements BackPropagationConnectionCalculator
	{
		private static final long serialVersionUID = 1L;

		protected float[] input;
		protected final int inputStartIndex;
		protected final int inputFeatureMaps;
		protected final int inputFeatureMapsLength;
		protected final int inputFeatureMapsDistance;
		protected final int inputMiniBatchDistance;
		protected final int miniBatchSize;

		protected float[] output;
		protected final int outputStartIndex;

		protected float[] cache;

		protected float[] ffActivations;
		protected final int activationsStartIndex;
		protected final int activationsFeatureMapsLength;
		protected final int activationsFeatureMapsDistance;
		protected final int activationsMiniBatchDistance;

		private final int n;
		private final float a;
		private final float b;

		public BPLRNKernel(ValuesProvider valuesProvider, ValuesProvider activationsProvider, RepeaterConnection connection, Layer targetLayer, float[] cache, int n, float a, float b)
		{
			Tensor input = TensorFactory.tensor(Util.getOppositeLayer(connection, targetLayer), connection, valuesProvider);
			Tensor output = TensorFactory.tensor(targetLayer, connection, valuesProvider);

			this.input = input.getElements();
			this.inputStartIndex = input.getStartIndex();
			this.inputFeatureMaps = input.getDimensions()[1];
			this.inputFeatureMapsLength = input.getDimensions()[2] * input.getDimensions()[3];
			this.inputMiniBatchDistance = input.getDimensionElementsDistance(0);
			this.inputFeatureMapsDistance = input.getDimensionElementsDistance(1);
			this.miniBatchSize = input.getDimensions()[0];

			this.output = output.getElements();
			this.outputStartIndex = output.getStartIndex();

			this.cache = cache;

			Tensor activations = activationsProvider.get(targetLayer);
			this.ffActivations = activations.getElements();
			this.activationsStartIndex = activations.getStartIndex();
			this.activationsFeatureMapsLength = activations.getDimensions()[2] * input.getDimensions()[3];
			this.activationsFeatureMapsDistance = activations.getDimensionElementsDistance(1);
			this.activationsMiniBatchDistance = activations.getDimensionElementsDistance(0);

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
			int inputStartId = inputStartIndex + startFM * inputFeatureMapsDistance + (id % inputFeatureMapsLength);
			int activationStartId = activationsStartIndex + startFM * activationsFeatureMapsDistance + (id % activationsFeatureMapsLength);
			int inputStart = 0;
			int activationStart = 0;
			float current = 0;

			for (int i = 0; i < miniBatchSize; i++)
			{
				current = 0;
				inputStart = inputStartId + inputMiniBatchDistance * i;
				activationStart = activationStartId + activationsMiniBatchDistance * i;
				for (int j = 0; j < fmCount; j++)
				{
					current += input[inputStart + j * inputFeatureMapsDistance] * ffActivations[activationStart + j * activationsFeatureMapsDistance];
				}

				arrId = id + i * inputMiniBatchDistance;

				float c = cache[arrId];

				output[outputStartIndex + arrId] = pow(c, -b) * (input[inputStartIndex + arrId] - (2 * a * b * current * ffActivations[activationsStartIndex + arrId]) / c);
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

		public int getInputMiniBatchDistance()
		{
			return inputMiniBatchDistance;
		}

		public int getMiniBatchSize()
		{
			return miniBatchSize;
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

		public float[] getFFActivations()
		{
			return ffActivations;
		}

		public int getActivationsStartIndex()
		{
			return activationsStartIndex;
		}

		public int getActivationsFeatureMapsLength()
		{
			return activationsFeatureMapsLength;
		}

		public int getActivationsFeatureMapsDistance()
		{
			return activationsFeatureMapsDistance;
		}

		public int getActivationsMiniBatchDistance()
		{
			return activationsMiniBatchDistance;
		}

		public int getInputFeatureMapsDistance()
		{
			return inputFeatureMapsDistance;
		}

		public int getN()
		{
			return n;
		}

		public float getB()
		{
			return b;
		}

		public float getA()
		{
			return a;
		}

		@Override
		public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
		{
			Tensor input = TensorFactory.tensor(Util.getOppositeLayer(connections.get(0), targetLayer), connections.get(0), valuesProvider);

			Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy()
					.execute(this, input.getDimensions()[1] * input.getDimensions()[2] * input.getDimensions()[3]);
		}

		@Override
		public ValuesProvider getActivations()
		{
			return null;
		}

		@Override
		public void setActivations(ValuesProvider activations)
		{
		}
	}
}
