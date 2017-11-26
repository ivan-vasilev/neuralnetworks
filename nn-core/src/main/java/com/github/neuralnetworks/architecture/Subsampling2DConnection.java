package com.github.neuralnetworks.architecture;

/**
 * Subsampling connections. Contains information about the size of the
 * subsampling region
 */
public class Subsampling2DConnection extends ConnectionsImpl
{

	private static final long serialVersionUID = 1L;

	protected int inputFeatureMapColumns;
	protected int inputFeatureMapRows;
	protected int filters;
	protected int subsamplingRegionRows;
	protected int subsamplingRegionColumns;
	protected int rowStride;
	protected int columnStride;
	protected int outputRowPadding;
	protected int outputColumnPadding;

	public Subsampling2DConnection(Layer inputLayer, Layer outputLayer, int inputFeatureMapRows, int inputFeatureMapColumns, int subsamplingRegionRows, int subsamplingRegionColumns, int filters, int rowStride, int columnStride, int outputRowPadding, int outputColumnPadding)
	{
		super(inputLayer, outputLayer);
		this.inputFeatureMapRows = inputFeatureMapRows;
		this.inputFeatureMapColumns = inputFeatureMapColumns;
		this.filters = filters;
		this.subsamplingRegionRows = subsamplingRegionRows;
		this.subsamplingRegionColumns = subsamplingRegionColumns;
		this.rowStride = rowStride;
		this.columnStride = columnStride;
		this.outputRowPadding = outputRowPadding;
		this.outputColumnPadding = outputColumnPadding;
	}

	public int getSubsamplingRegionRows()
	{
		return subsamplingRegionRows;
	}

	public int getSubsamplingRegionCols()
	{
		return subsamplingRegionColumns;
	}

	public int getSubsamplingRegionLength()
	{
		return getSubsamplingRegionRows() * getSubsamplingRegionCols();
	}

	@Override
	public int getInputUnitCount()
	{
		return inputFeatureMapRows * inputFeatureMapColumns * filters;
	}

	@Override
	public int getOutputUnitCount()
	{
		return getOutputFeatureMapLength() * filters;
	}

	public int getOutputUnitCountWithPadding()
	{
		return getOutputFeatureMapLengthWithPadding() * filters;
	}

	public int getInputFeatureMapColumns()
	{
		return inputFeatureMapColumns;
	}

	public void setInputFeatureMapColumns(int inputFeatureMapColumns)
	{
		this.inputFeatureMapColumns = inputFeatureMapColumns;
	}

	public int getInputFeatureMapRows()
	{
		return inputFeatureMapRows;
	}

	public void setInputFeatureMapRows(int inputFeatureMapRows)
	{
		this.inputFeatureMapRows = inputFeatureMapRows;
	}

	public int getOutputFeatureMapRows()
	{
		return (inputFeatureMapRows - subsamplingRegionRows) / rowStride + 1;
	}

	public int getOutputFeatureMapRowsWithPadding()
	{
		return getOutputFeatureMapRows() + 2 * outputRowPadding;
	}

	public int getOutputFeatureMapColumns()
	{
		return (inputFeatureMapColumns - subsamplingRegionColumns) / columnStride + 1;
	}
	
	public int getOutputFeatureMapColumnsWithPadding()
	{
		return getOutputFeatureMapColumns() + 2 * outputColumnPadding;
	}

	public int getFilters()
	{
		return filters;
	}

	public void setFilters(int filters)
	{
		this.filters = filters;
	}

	public int getInputFeatureMapLength()
	{
		return inputFeatureMapRows * inputFeatureMapColumns;
	}

	public int getOutputFeatureMapLength()
	{
		return getOutputFeatureMapRows() * getOutputFeatureMapColumns();
	}

	public int getOutputFeatureMapLengthWithPadding()
	{
		return getOutputFeatureMapRowsWithPadding() * getOutputFeatureMapColumnsWithPadding();
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
