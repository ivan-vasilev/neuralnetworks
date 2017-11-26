package com.github.neuralnetworks.architecture;

import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;

/**
 * Convolutional connection between layers (for 2d input data)
 */
public class Conv2DConnection extends ConnectionsImpl implements WeightsConnections
{

	private static final long serialVersionUID = 1L;

	/**
	 * The list of filters to be used in the connection
	 */
	protected Tensor weights;
	protected int inputFeatureMapColumns;
	protected int inputFeatureMapRows;
	protected int rowStride;
	protected int columnStride;
	protected int outputRowPadding;
	protected int outputColumnPadding;

	public Conv2DConnection(Layer inputLayer, Layer outputLayer, int inputFeatureMapColumns, int inputFeatureMapRows, int inputFilters, int filterRows, int filterColumns, int outputFilters, int rowStride, int columnStride, int outputRowPadding, int outputColumnPadding)
	{
		super(inputLayer, outputLayer);
		this.inputFeatureMapColumns = inputFeatureMapColumns;
		this.inputFeatureMapRows = inputFeatureMapRows;
		this.rowStride = rowStride;
		this.columnStride = columnStride;
		this.weights = TensorFactory.tensor(outputFilters, inputFilters, filterRows, filterColumns);
		this.outputRowPadding = outputRowPadding;
		this.outputColumnPadding = outputColumnPadding;
	}

	public Conv2DConnection(Layer inputLayer, Layer outputLayer, int inputFeatureMapRows, int inputFeatureMapColumns, Tensor weights, int rowStride, int columnStride, int outputRowPadding, int outputColumnPadding)
	{
		super(inputLayer, outputLayer);
		this.inputFeatureMapRows = inputFeatureMapRows;
		this.inputFeatureMapColumns = inputFeatureMapColumns;
		this.rowStride = rowStride;
		this.columnStride = columnStride;
		this.weights = weights;
		this.outputRowPadding = outputRowPadding;
		this.outputColumnPadding = outputColumnPadding;
	}

	@Override
	public Tensor getWeights()
	{
		return weights;
	}

	public void setWeights(Tensor weights)
	{
		this.weights = weights;
	}

	public int getFilterRows()
	{
		return weights.getDimensions()[2];
	}

	public int getFilterColumns()
	{
		return weights.getDimensions()[3];
	}

	@Override
	public int getInputUnitCount()
	{
		return inputFeatureMapRows * inputFeatureMapColumns * weights.getDimensions()[1];
	}

	@Override
	public int getOutputUnitCount()
	{
		return getOutputFeatureMapLength() * weights.getDimensions()[0];
	}

	public int getOutputUnitCountWithPadding()
	{
		return getOutputFeatureMapLengthWithPadding() * weights.getDimensions()[0];
	}

	public int getInputFeatureMapColumns()
	{
		return inputFeatureMapColumns;
	}

	public int getInputFeatureMapRows()
	{
		return inputFeatureMapRows;
	}

	public int getInputFeatureMapLength()
	{
		return inputFeatureMapRows * inputFeatureMapColumns;
	}

	public int getInputFilters()
	{
		return weights.getDimensions()[1];
	}

	public int getOutputFeatureMapRows()
	{
		return (inputFeatureMapRows - weights.getDimensions()[2]) / rowStride + 1;
	}

	public int getOutputFeatureMapRowsWithPadding()
	{
		return getOutputFeatureMapRows() + 2 * outputRowPadding;
	}

	public int getOutputFeatureMapColumns()
	{
		return (inputFeatureMapColumns - weights.getDimensions()[3]) / columnStride + 1;
	}
	
	public int getOutputFeatureMapColumnsWithPadding()
	{
		return getOutputFeatureMapColumns() + 2 * outputColumnPadding;
	}

	public int getOutputFeatureMapLength()
	{
		return getOutputFeatureMapRows() * getOutputFeatureMapColumns();
	}

	public int getOutputFeatureMapLengthWithPadding()
	{
		return getOutputFeatureMapRowsWithPadding() * getOutputFeatureMapColumnsWithPadding();
	}
	
	public int getOutputFilters()
	{
		return weights.getDimensions()[0];
	}

	public int getRowStride()
	{
		return rowStride;
	}

	public int getColumnStride()
	{
		return columnStride;
	}

	public int getOutputRowPadding()
	{
		return outputRowPadding;
	}

	public int getOutputColumnPadding()
	{
		return outputColumnPadding;
	}
}
