package com.github.neuralnetworks.calculation.operations.aparapi;

import java.util.Arrays;
import java.util.List;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Util;

/**
 * Base class for convolutional operations (2d)
 * This connection accept as input a single training example (as opposed to the weighted sum which works with multiple).
 *
 * !!! IMPORTANT !!!
 * Aparapi only works one-dimensional arrays of primitive data types can only call member methods of the Kernel class itself.
 */
public abstract class AparapiConv2D extends Kernel implements ConnectionCalculator
{

	private static final long serialVersionUID = 1L;

	/**
	 * input
	 */
	protected float[] input;
	protected final int inputStartIndex;
	protected final int inputFeatureMapRowsDistance;
	protected final int inputFeatureMapColumnsDistance;
	protected final int inputMiniBatchDistance;

	/**
	 * output
	 */
	protected float[] output;
	protected final int outputStartIndex;
	protected final int outputFeatureMapsDistance;
	protected final int outputFeatureMapRowsDistance;
	protected final int outputFeatureMapColumnsDistance;
	protected final int outputMiniBatchDistance;
	protected final int outputFeatureMapLength; // output columns * output rows
	protected final int outputColumns; // output column count (columns of image for example)

	/**
	 * combined feature weights of all feature maps
	 */
	// @Local TODO
	protected final float[] weights;
	protected final int weightsStartIndex;

	/**
	 * weights for single feature map
	 */
	protected final int featureMapWeights;

	/**
	 * input offset for each feature map in respect to the start index
	 */
	// @Local TODO
	@Constant
	protected final int[] featureMapOffsets;

	/**
	 * number of samples per calculation (for example number of images)
	 */
	protected final int miniBatchSize;

	/**
	 * stride
	 */
	protected final int rowStride;
	protected final int columnStride;

	public AparapiConv2D(Conv2DConnection c, ValuesProvider valuesProvider, Layer targetLayer)
	{
		super();

		Tensor input = null, output = null;
		if (targetLayer == c.getOutputLayer())
		{
			input = TensorFactory.tensor(Util.getOppositeLayer(c, targetLayer), c, valuesProvider);
			output = TensorFactory.tensor(targetLayer, c, valuesProvider);
		} else
		{
			input = TensorFactory.tensor(targetLayer, c, valuesProvider);
			output = TensorFactory.tensor(Util.getOppositeLayer(c, targetLayer), c, valuesProvider);
		}

		this.input = input.getElements();
		this.inputStartIndex = input.getStartIndex();
		this.inputMiniBatchDistance = input.getDimensionElementsDistance(0);
		this.inputFeatureMapRowsDistance = input.getDimensionElementsDistance(2);
		this.inputFeatureMapColumnsDistance = input.getDimensionElementsDistance(3);

		this.output = output.getElements();
		this.outputStartIndex = output.getStartIndex();
		this.outputMiniBatchDistance = output.getDimensionElementsDistance(0);
		this.outputFeatureMapsDistance = output.getDimensionElementsDistance(1);
		this.outputFeatureMapRowsDistance = output.getDimensionElementsDistance(2);
		this.outputFeatureMapColumnsDistance = output.getDimensionElementsDistance(3);

		this.weights = c.getWeights().getElements();
		this.weightsStartIndex = c.getWeights().getStartIndex();

		this.miniBatchSize = TensorFactory.batchSize(valuesProvider);
		this.outputColumns = c.getOutputFeatureMapColumns();
		this.outputFeatureMapLength = c.getOutputFeatureMapLength();
		this.rowStride = c.getRowStride();
		this.columnStride = c.getColumnStride();
		this.featureMapWeights = c.getFilterColumns() * c.getFilterRows() * c.getInputFilters();
		this.featureMapOffsets = new int[featureMapWeights];

		int inputFeatureMapsDistance = input.getDimensionElementsDistance(1);

		for (int i = 0, offset = 0; i < c.getInputFilters(); i++)
		{
			for (int j = 0; j < c.getFilterRows(); j++)
			{
				for (int k = 0; k < c.getFilterColumns(); k++)
				{
					featureMapOffsets[offset++] = i * inputFeatureMapsDistance + j * inputFeatureMapRowsDistance + k * inputFeatureMapColumnsDistance;
				}
			}
		}
	}


	@Override
	public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		if (accept(connections, valuesProvider))
		{
			Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(this, targetLayer.getUnitCount(connections));
		} else
		{
			throw new IllegalArgumentException("A parameter does not match");
		}
	}

	public boolean accept(List<Connections> connections, ValuesProvider valuesProvider)
	{
		if (connections.size() != 1 || !(connections.get(0) instanceof Conv2DConnection)) {
			return false;
		}

		Conv2DConnection c = (Conv2DConnection) connections.get(0);

		if (TensorFactory.batchSize(valuesProvider) != miniBatchSize)
		{
			return false;
		}

		if (TensorFactory.tensor(c.getOutputLayer(), c, valuesProvider).getElements() != output)
		{
			return false;
		}

		if (TensorFactory.tensor(Util.getOppositeLayer(c, c.getOutputLayer()), c, valuesProvider).getElements() != input)
		{
			return false;
		}

		return true;
	}


	@SuppressWarnings("unused")
	public void calculate(Conv2DConnection c, ValuesProvider valuesProvider, Layer targetLayer)
	{
		if (c != null)
		{
			Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(this, targetLayer.getUnitCount(Arrays.asList(new Conv2DConnection[] { c })));
		}
	}

	@Override
	public void run()
	{
		int id = getGlobalId();

		conv(weightsStartIndex + featureMapWeights * (id / outputFeatureMapLength),
				inputStartIndex + ((id % outputFeatureMapLength) / outputColumns) * inputFeatureMapRowsDistance * rowStride + (id % outputColumns) * inputFeatureMapColumnsDistance * columnStride,
				outputStartIndex + (id / outputFeatureMapLength) * outputFeatureMapsDistance + ((id % outputFeatureMapLength) / outputColumns) * outputFeatureMapRowsDistance + (id % outputColumns)
						* outputFeatureMapColumnsDistance);
	}

	/**
	 * the actual convolution
	 * 
	 * @param weightsStartId
	 * @param inputStartId
	 * @param outputStartId
	 */
	@SuppressWarnings("unused")
	protected void conv(int weightsStartId, int inputStartId, int outputStartId)
	{
	}

	public float[] getInput()
	{
		return input;
	}

	public int getInputStartIndex()
	{
		return inputStartIndex;
	}

	public int getInputFeatureMapRowsDistance()
	{
		return inputFeatureMapRowsDistance;
	}

	public int getInputFeatureMapColumnsDistance()
	{
		return inputFeatureMapColumnsDistance;
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

	public int getOutputMiniBatchDistance()
	{
		return outputMiniBatchDistance;
	}

	public int getOutputFeatureMapLength()
	{
		return outputFeatureMapLength;
	}


	public int getOutputColumns()
	{
		return outputColumns;
	}

	public float[] getWeights()
	{
		return weights;
	}

	public int getWeightsStartIndex()
	{
		return weightsStartIndex;
	}

	public int getFeatureMapWeights()
	{
		return featureMapWeights;
	}

	public int[] getFeatureMapOffsets()
	{
		return featureMapOffsets;
	}

	public int getMiniBatchSize()
	{
		return miniBatchSize;
	}

	public int getRowStride()
	{
		return rowStride;
	}

	public int getColumnStride()
	{
		return columnStride;
	}
}
