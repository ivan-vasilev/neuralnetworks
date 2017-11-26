package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.backpropagation.WeightUpdates;
import com.github.neuralnetworks.util.Environment;

/**
 * Base Aparapi connection calculator for subsampling layers.
 * 
 * !!! IMPORTANT !!!
 * Aparapi only works one-dimensional arrays of primitive data types can only call member methods of the Kernel class itself.
 */
public class AparapiBackpropagationConv2DWeightUpdates extends Kernel implements WeightUpdates
{
	private static final long serialVersionUID = 8931101094464503687L;

	/**
	 * input samples count
	 */
	protected final int miniBatchSize;


	/**
	 * weights
	 */
	protected final Tensor weightsTensor;
	protected final float[] weights;
	protected final int weightsStartIndex;
	protected final int weightsOutputFiltersDistance;
	protected final int weightsInputFiltersDistance;
	protected final int weightsRowsDistance;
	protected final int weightsColumnsDistance;
	protected final int weightsfilterRows;
	protected final int weightsfilterColumns;

	protected final float[] weightsUpdates;
	protected final int weightsUpdatesOutputFiltersDistance;
	protected final int weightsUpdatesInputFiltersDistance;
	protected final int weightsUpdatesRowsDistance;

	/**
	 * Stride
	 */
	protected final int columnStride;
	protected final int rowStride;

	/**
	 * activations data
	 */
	protected float[] activation;
	protected final int activationStartIndex;
	protected final int activationFeatureMapsDistance;
	protected final int activationFeatureMapRowsDistance;
	protected final int activationFeatureMapColumnsDistance;
	protected final int activationFeatureMaps;
	protected final int activationMiniBatchDistance;

	/**
	 * output
	 */
	protected Tensor outputTensor;
	protected float[] output;
	protected final int outputStartIndex;
	protected final int outputFeatureMapsDistance;
	protected final int outputFeatureMapRowsDistance;
	protected final int outputFeatureMapColumnsDistance;
	protected final int outputFeatureMapRows;
	protected final int outputFeatureMapColumns;
	protected final int outputFeatureMaps;
	protected final int outputMiniBatchDistance;

	/**
	 * BP parameters
	 */
	protected float learningRate;
	protected float momentum;
	protected float l1weightDecay;
	protected float l2weightDecay;

	//protected Conv2DConnection c;

	public AparapiBackpropagationConv2DWeightUpdates(Conv2DConnection c, ValuesProvider valuesProvider, ValuesProvider activations, Tensor weightUpdates)
	{
		super();

		//this.c = c;
		Tensor activation = TensorFactory.tensor(c.getInputLayer(), c, activations);
		Tensor output = TensorFactory.tensor(c.getOutputLayer(), c, valuesProvider);

		this.activation = activation.getElements();
		this.activationStartIndex = activation.getStartIndex();
		this.activationMiniBatchDistance = activation.getDimensionElementsDistance(0);
		this.activationFeatureMapsDistance = activation.getDimensionElementsDistance(1);
		this.activationFeatureMapRowsDistance = activation.getDimensionElementsDistance(2);
		this.activationFeatureMapColumnsDistance = activation.getDimensionElementsDistance(3);
		this.activationFeatureMaps = c.getWeights().getDimensions()[1];

		this.outputTensor = output;
		this.output = output.getElements();
		this.outputStartIndex = output.getStartIndex();
		this.outputMiniBatchDistance = output.getDimensionElementsDistance(0);
		this.outputFeatureMapsDistance = output.getDimensionElementsDistance(1);
		this.outputFeatureMapRowsDistance = output.getDimensionElementsDistance(2);
		this.outputFeatureMapColumnsDistance = output.getDimensionElementsDistance(3);
		this.outputFeatureMapRows = c.getOutputFeatureMapRows();
		this.outputFeatureMapColumns = c.getOutputFeatureMapColumns();
		this.outputFeatureMaps = c.getWeights().getDimensions()[0];

		this.miniBatchSize = activation.getDimensions()[0];

		this.rowStride = c.getRowStride();
		this.columnStride = c.getColumnStride();

		this.weightsTensor = c.getWeights();
		this.weights = c.getWeights().getElements();
		this.weightsStartIndex = c.getWeights().getStartIndex();
		this.weightsOutputFiltersDistance = c.getWeights().getDimensionElementsDistance(0);
		this.weightsInputFiltersDistance = c.getWeights().getDimensionElementsDistance(1);
		this.weightsRowsDistance = c.getWeights().getDimensionElementsDistance(2);
		this.weightsColumnsDistance = c.getWeights().getDimensionElementsDistance(3);
		this.weightsfilterRows = c.getWeights().getDimensions()[2];
		this.weightsfilterColumns = c.getWeights().getDimensions()[3];

		this.weightsUpdates = weightUpdates.getElements();
		this.weightsUpdatesOutputFiltersDistance = weightUpdates.getDimensionElementsDistance(0);
		this.weightsUpdatesInputFiltersDistance = weightUpdates.getDimensionElementsDistance(1);
		this.weightsUpdatesRowsDistance = weightUpdates.getDimensionElementsDistance(2);
	}

	@Override
	public void updateWeights(float learningRate, float momentum, float l1weightDecay, float l2weightDecay)
	{
		this.learningRate = learningRate;
		this.momentum = momentum;
		this.l1weightDecay = l1weightDecay;
		this.l2weightDecay = l2weightDecay;

		Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(this, weightsTensor.getSize());
//		if (CustomArrays.getInstance().getCustomArrays().get(c).size() != 0) {
//			throw new RuntimeException("" + CustomArrays.getInstance().getCustomArrays().get(c).size());
//		}
	}

	@Override
	public void run()
	{
		int id = getGlobalId();
		int outputFM = id / weightsOutputFiltersDistance;
		int inputFM = (id % weightsOutputFiltersDistance) / weightsInputFiltersDistance;
		int row = (id % weightsInputFiltersDistance) / weightsRowsDistance;
		int col = (id % weightsRowsDistance) / weightsColumnsDistance;

		float weightUpdate = 0, weight = 0;
		int weightUpdateIndex = 0;
		int outputStart = outputStartIndex + outputFM  * outputFeatureMapsDistance;
		int activationStart = activationStartIndex + inputFM * activationFeatureMapsDistance;

		for (int m = 0; m < miniBatchSize; m++)
		{
			for (int i = 0; i < outputFeatureMapRows; i++)
			{
				for (int j = 0; j < outputFeatureMapColumns; j++)
				{
					weightUpdate += output[outputStart + i * outputFeatureMapRowsDistance + j * outputFeatureMapColumnsDistance + m * outputMiniBatchDistance] *
							activation[activationStart + (i * rowStride + row) * activationFeatureMapRowsDistance + (j * columnStride + col) * activationFeatureMapColumnsDistance + m * activationMiniBatchDistance];

//					CustomArray ca = new CustomArray(new int[] {
//							activationStart + (i * rowStride + row) * activationFeatureMapRowsDistance + (j * columnStride + col) * activationFeatureMapColumnsDistance + m * activationMiniBatchDistance,
//							weightsStartIndex + outputFM * weightsOutputFiltersDistance + inputFM * weightsInputFiltersDistance + row * weightsRowsDistance + col * weightsColumnsDistance,
//							outputStart + i * outputFeatureMapRowsDistance + j * outputFeatureMapColumnsDistance + m * outputMiniBatchDistance		
//					});
//
//					if (!CustomArrays.getInstance().getCustomArrays().get(c).contains(ca)) {
//						throw new RuntimeException(Arrays.toString(ca.getArray()));
//					} else {
//						CustomArrays.getInstance().getCustomArrays().get(c).remove(ca);
//					}
				}
			}
		}

		weightUpdateIndex = outputFM * weightsUpdatesOutputFiltersDistance + inputFM * weightsUpdatesInputFiltersDistance + row * weightsUpdatesRowsDistance + col;
		weight = weights[weightsStartIndex + outputFM * weightsOutputFiltersDistance + inputFM * weightsInputFiltersDistance + row * weightsRowsDistance + col * weightsColumnsDistance];
		weightUpdate = (learningRate * weightUpdate) / miniBatchSize + momentum * weightsUpdates[weightUpdateIndex] - learningRate * l1weightDecay * weight - l2weightDecay * weight * weight;

		weights[weightsStartIndex + outputFM * weightsOutputFiltersDistance + inputFM * weightsInputFiltersDistance + row * weightsRowsDistance + col * weightsColumnsDistance] += weightUpdate;
		weightsUpdates[weightUpdateIndex] = weightUpdate;
	}

	public int getMiniBatchSize()
	{
		return miniBatchSize;
	}

	public float[] getWeights()
	{
		return weights;
	}

	public int getWeightsStartIndex()
	{
		return weightsStartIndex;
	}

	public int getWeightsOutputFiltersDistance()
	{
		return weightsOutputFiltersDistance;
	}

	public int getWeightsInputFiltersDistance()
	{
		return weightsInputFiltersDistance;
	}

	public int getColumnStride()
	{
		return columnStride;
	}

	public int getRowStride()
	{
		return rowStride;
	}

	public float[] getActivation()
	{
		return activation;
	}

	public int getActivationStartIndex()
	{
		return activationStartIndex;
	}

	public int getActivationFeatureMapsDistance()
	{
		return activationFeatureMapsDistance;
	}

	public int getActivationFeatureMapRowsDistance()
	{
		return activationFeatureMapRowsDistance;
	}

	public int getActivationFeatureMapColumnsDistance()
	{
		return activationFeatureMapColumnsDistance;
	}

	public int getActivationMiniBatchDistance()
	{
		return activationMiniBatchDistance;
	}

	public float[] getOutput()
	{
		return output;
	}

	public int getOutputStartIndex()
	{
		return outputStartIndex;
	}

	public int getOutputFeatureMapsDistance()
	{
		return outputFeatureMapsDistance;
	}

	public int getOutputFeatureMapRowsDistance()
	{
		return outputFeatureMapRowsDistance;
	}

	public int getOutputFeatureMapColumnsDistance()
	{
		return outputFeatureMapColumnsDistance;
	}

	public int getOutputFeatureMapColumns()
	{
		return outputFeatureMapColumns;
	}

	public int getOutputMiniBatchDistance()
	{
		return outputMiniBatchDistance;
	}

	public int getWeightsfilterRows()
	{
		return weightsfilterRows;
	}

	public int getWeightsfilterColumns()
	{
		return weightsfilterColumns;
	}

	public float[] getWeightsUpdates()
	{
		return weightsUpdates;
	}

	public int getActivationFeatureMaps()
	{
		return activationFeatureMaps;
	}

	public int getOutputFeatureMaps()
	{
		return outputFeatureMaps;
	}

	public float getLearningRate()
	{
		return learningRate;
	}

	public void setLearningRate(float learningRate)
	{
		this.learningRate = learningRate;
	}

	public float getMomentum()
	{
		return momentum;
	}

	public void setMomentum(float momentum)
	{
		this.momentum = momentum;
	}

	public float getL1weightDecay()
	{
		return l1weightDecay;
	}

	public void setL1weightDecay(float l1weightDecay)
	{
		this.l1weightDecay = l1weightDecay;
	}

	public float getL2weightDecay()
	{
		return l2weightDecay;
	}

	public void setL2weightDecay(float l2weightDecay)
	{
		this.l2weightDecay = l2weightDecay;
	}

	public ValuesProvider getActivations()
	{
		return null;
	}

	public int getWeightsRowsDistance()
	{
		return weightsRowsDistance;
	}

	public int getWeightsColumnsDistance()
	{
		return weightsColumnsDistance;
	}

	public int getOutputFeatureMapRows()
	{
		return outputFeatureMapRows;
	}

	public int getWeightsUpdatesOutputFiltersDistance()
	{
		return weightsUpdatesOutputFiltersDistance;
	}

	public int getWeightsUpdatesInputFiltersDistance()
	{
		return weightsUpdatesInputFiltersDistance;
	}

	public int getWeightsUpdatesRowsDistance()
	{
		return weightsUpdatesRowsDistance;
	}

	public Tensor getOutputTensor()
	{
		return outputTensor;
	}

	@SuppressWarnings("unused")
	public void setActivations(ValuesProvider activations)
	{
	}
}
