package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import java.util.List;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Util;

/**
 * Base Aparapi connection calculator for subsampling layers.
 * 
 * !!! IMPORTANT !!!
 * Aparapi only works one-dimensional arrays of primitive data types can only call member methods of the Kernel class itself.
 */
public class AparapiBackpropagationSubsampling2D2 extends Kernel implements BackPropagationConnectionCalculator
{

	private static final long serialVersionUID = 8931101094464503687L;

	/**
	 * input samples count
	 */
	protected final int miniBatchSize;

	/**
	 * subsampling region rows
	 */
	protected final int subsamplingRows;

	/**
	 * subsampling region columns
	 */
	protected final int subsamplingCols;

	/**
	 * Length of the subsampling region (subsampling rows * subsampling cols)
	 */
	protected final int regionLength;

	/**
	 * offset from start when mapping input to output
	 */
	protected final int ioRowsOffset;

	/**
	 * offset from start when mapping input to output
	 */
	protected final int ioColumnsOffset;

	/**
	 * Stride
	 */
	protected final int columnStride;
	protected final int rowStride;

	/**
	 * Contains the offset in the input array for each cell of the current region. The offset is calculated in respect to the first cell of the region
	 */
	// @Local TODO
	protected final int[] featureMapOffsets;

	/**
	 * input data
	 */
	protected float[] input;
	protected final int inputStartIndex;
	protected final int inputFeatureMapsDistance;
	protected final int inputFeatureMapRowsDistance;
	protected final int inputFeatureMapColumnsDistance;
	protected final int inputFeatureMapLength;
	protected final int inputFeatureMapRows;
	protected final int inputFeatureMapColumns;
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

	protected float[] ffActivation;

	public AparapiBackpropagationSubsampling2D2(Subsampling2DConnection c, ValuesProvider valuesProvider, ValuesProvider activations)
	{
		super();

		Tensor input = TensorFactory.tensor(c.getInputLayer(), c, valuesProvider);
		Tensor output = TensorFactory.tensor(c.getOutputLayer(), c, valuesProvider);

		this.input = input.getElements();
		this.inputStartIndex = input.getStartIndex();
		this.inputMiniBatchDistance = input.getDimensionElementsDistance(0);
		this.inputFeatureMapsDistance = input.getDimensionElementsDistance(1);
		this.inputFeatureMapRowsDistance = input.getDimensionElementsDistance(2);
		this.inputFeatureMapColumnsDistance = input.getDimensionElementsDistance(3);
		this.inputFeatureMapLength = c.getInputFeatureMapLength();
		this.inputFeatureMapRows = c.getInputFeatureMapRows();
		this.inputFeatureMapColumns = c.getInputFeatureMapColumns();

		this.output = output.getElements();
		this.outputStartIndex = output.getStartIndex();
		this.outputMiniBatchDistance = output.getDimensionElementsDistance(0);
		this.outputFeatureMapsDistance = output.getDimensionElementsDistance(1);
		this.outputFeatureMapRowsDistance = output.getDimensionElementsDistance(2);
		this.outputFeatureMapColumnsDistance = output.getDimensionElementsDistance(3);

		this.miniBatchSize = input.getDimensions()[0];

		this.subsamplingRows = c.getSubsamplingRegionRows();
		this.subsamplingCols = c.getSubsamplingRegionCols();
		this.regionLength = c.getSubsamplingRegionLength();
		this.ioRowsOffset = ((c.getInputFeatureMapRows() - c.getSubsamplingRegionRows()) % c.getRowStride()) / 2;
		this.ioColumnsOffset = ((c.getInputFeatureMapColumns() - c.getSubsamplingRegionCols()) % c.getColumnStride()) / 2;
		this.rowStride = c.getRowStride();
		this.columnStride = c.getColumnStride();

		this.featureMapOffsets = new int[regionLength];
		for (int j = 0, offset = 0; j < subsamplingRows; j++)
		{
			for (int k = 0; k < subsamplingCols; k++)
			{
				featureMapOffsets[offset++] = j * inputFeatureMapRowsDistance + k * inputFeatureMapColumnsDistance;
			}
		}

		ffActivation = TensorFactory.tensor(c.getInputLayer(), c, activations).getElements();
	}

	@Override
	public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		if (connections.size() > 0)
		{
			Subsampling2DConnection c = (Subsampling2DConnection) connections.get(0);
			if (targetLayer == c.getInputLayer())
			{
				Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(this, targetLayer.getUnitCount(connections));
			} else
			{
				Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(this, Util.getOppositeLayer(c, targetLayer).getUnitCount(connections));
			}
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.amd.aparapi.Kernel#run()
	 * input start index is calculated and passed to the pooling method
	 */
	@Override
	public void run()
	{
		int id = getGlobalId();

		// get current offset
		int fm = id / inputFeatureMapLength;
		int fmOffset = id % inputFeatureMapLength;
		int fmRow = fmOffset / inputFeatureMapColumns + 1;
		int fmCol = fmOffset % inputFeatureMapColumns + 1;

		for (int m = 0; m < miniBatchSize; m++)
		{
			input[inputStartIndex + id + m * inputMiniBatchDistance] = 0;

			for (int i = (int) max(0, ceil((fmRow - subsamplingRows) / ((float) rowStride)) * rowStride); i < min(inputFeatureMapRows - subsamplingRows + 1, fmRow); i += rowStride)
			{
				for (int j = (int) max(0, ceil((fmCol - subsamplingCols) / ((float) columnStride)) * columnStride); j < min(inputFeatureMapColumns - subsamplingCols + 1, fmCol); j += columnStride)
				{
					pool(inputStartIndex + fm * inputFeatureMapsDistance + (ioRowsOffset + i) * inputFeatureMapRowsDistance + (ioColumnsOffset + j) * inputFeatureMapColumnsDistance,
							outputStartIndex + fm * outputFeatureMapsDistance + (i / rowStride) * outputFeatureMapRowsDistance + (j / columnStride) * outputFeatureMapColumnsDistance, m);
				}
			}
		}
	}

	public boolean accept(Subsampling2DConnection c, ValuesProvider valuesProvider)
	{
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

	/**
	 * This is where the subsampling happens
	 */
	@SuppressWarnings("unused")
	protected void pool(int inputFilterStartIndex, int outputIndex, int sample)
	{
	}

	public int getMiniBatchSize()
	{
		return miniBatchSize;
	}

	public int getSubsamplingRows()
	{
		return subsamplingRows;
	}

	public int getSubsamplingCols()
	{
		return subsamplingCols;
	}

	public int getRegionLength()
	{
		return regionLength;
	}

	public int getIoRowsOffset()
	{
		return ioRowsOffset;
	}

	public int getIoColumnsOffset()
	{
		return ioColumnsOffset;
	}

	public int getColumnStride()
	{
		return columnStride;
	}

	public int getRowStride()
	{
		return rowStride;
	}

	public int[] getFeatureMapOffsets()
	{
		return featureMapOffsets;
	}

	public float[] getInput()
	{
		return input;
	}

	public int getInputStartIndex()
	{
		return inputStartIndex;
	}

	public int getInputFeatureMapsDistance()
	{
		return inputFeatureMapsDistance;
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

	public int getInputFeatureMapLength()
	{
		return inputFeatureMapLength;
	}

	public int getInputFeatureMapRows()
	{
		return inputFeatureMapRows;
	}

	public int getInputFeatureMapColumns()
	{
		return inputFeatureMapColumns;
	}

	public float[] getFfActivation()
	{
		return ffActivation;
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
