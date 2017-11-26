package com.github.neuralnetworks.calculation.operations.aparapi.bp;

import java.util.List;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.operations.ClearValuesManager;
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
public class AparapiBackpropagationConv2D2 extends Kernel implements BackPropagationConnectionCalculator
{

	private static final long serialVersionUID = 8931101094464503687L;

	/**
	 * input samples count
	 */
	protected final int miniBatchSize;


	/**
	 * weights
	 */
	protected final float[] weights;
	protected final int weightsStartIndex;
	protected final int weightsOutputFiltersDistance;
	protected final int weightsInputFiltersDistance;
	protected final int weightsRowsDistance;
	protected final int weightsColumnsDistance;

	/**
	 * conv filter rows
	 */
	protected final int filterRows;

	/**
	 * conv filter columns
	 */
	protected final int filterCols;

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
	protected final int outputFeatureMaps;
	protected final int outputMiniBatchDistance;

	protected final AparapiBackpropagationConv2DWeightUpdates aparapiWeightUpdates;

	protected int clear;

	//protected Conv2DConnection c;

	public AparapiBackpropagationConv2D2(Conv2DConnection c, ValuesProvider valuesProvider, ValuesProvider activations, Tensor weightUpdates)
	{
		super();

		//this.c = c;
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
		this.outputFeatureMaps = c.getOutputFilters();

		this.miniBatchSize = input.getDimensions()[0];

		this.filterRows = c.getFilterRows();
		this.filterCols = c.getFilterColumns();
		this.ioRowsOffset = (c.getInputFeatureMapRows() % filterRows) / 2;
		this.ioColumnsOffset = (c.getInputFeatureMapColumns() % filterCols) / 2;
		this.rowStride = c.getRowStride();
		this.columnStride = c.getColumnStride();

		this.weights = c.getWeights().getElements();
		this.weightsStartIndex = c.getWeights().getStartIndex();
		this.weightsOutputFiltersDistance = c.getWeights().getDimensionElementsDistance(0);
		this.weightsInputFiltersDistance = c.getWeights().getDimensionElementsDistance(1);
		this.weightsRowsDistance = c.getWeights().getDimensionElementsDistance(2);
		this.weightsColumnsDistance = c.getWeights().getDimensionElementsDistance(3);

		this.aparapiWeightUpdates = new AparapiBackpropagationConv2DWeightUpdates(c, valuesProvider, activations, weightUpdates);

		if (!ClearValuesManager.getInstance().isCleared(input))
		{
			ClearValuesManager.getInstance().addToCleared(input);
			clear = 0;
		} else
		{
			clear = 1;
		}
	}

	@Override
	public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		if (connections.size() > 0)
		{
			Conv2DConnection c = (Conv2DConnection) connections.get(0);
			if (targetLayer == c.getInputLayer())
			{
				Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(this, targetLayer.getUnitCount(connections));

//				if (CustomArrays.getInstance().getCustomArrays().get(c).size() != 0) {
//					throw new RuntimeException("" + CustomArrays.getInstance().getCustomArrays().get(c).size());
//				}
			} else
			{
				Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(this, Util.getOppositeLayer(c, targetLayer).getUnitCount(connections));

//				if (CustomArrays.getInstance().getCustomArrays().get(c).size() != 0) {
//					throw new RuntimeException("" + CustomArrays.getInstance().getCustomArrays().get(c).size());
//				}
			}
		}
	}

	@Override
	public void run()
	{
		int id = getGlobalId();

		// get current offset
		int fm = id / inputFeatureMapLength;
		int fmOffset = id % inputFeatureMapLength;
		int fmRow = fmOffset / inputFeatureMapColumns + 1;
		int fmCol = fmOffset % inputFeatureMapColumns + 1;

		int inputIndex = inputStartIndex + fm * inputFeatureMapsDistance + (fmRow - 1) * inputFeatureMapRowsDistance + (fmCol - 1) * inputFeatureMapColumnsDistance;
		int rowStart = (int) max(0, ceil((fmRow - filterRows) / ((float) rowStride)));
		int rowEnd = min(inputFeatureMapRows - filterRows + 1, fmRow) / rowStride;
		int colStart = (int) max(0, ceil((fmCol - filterCols) / ((float) columnStride)));
		int colEnd = min(inputFeatureMapColumns - filterCols + 1, fmCol) / columnStride;

		float currentWeight = 0;
		int weightsRow = 0, weightsCol = 0, outputIndex = 0;

		for (int l = 0; l < miniBatchSize; l++)
		{
			input[inputIndex + l * inputMiniBatchDistance] *= clear; 

			for (int i = rowStart; i < rowEnd; i++)
			{
				for (int j = colStart; j < colEnd; j++)
				{
					weightsRow = fmRow - i * rowStride - 1;
					weightsCol = fmCol - j * columnStride - 1;

					for (int k = 0; k < outputFeatureMaps; k++)
					{
						outputIndex = outputStartIndex + k * outputFeatureMapsDistance + i * outputFeatureMapRowsDistance + j
								* outputFeatureMapColumnsDistance;
						currentWeight = weights[weightsStartIndex + k * weightsOutputFiltersDistance + fm * weightsInputFiltersDistance + weightsRow * weightsRowsDistance + weightsCol * weightsColumnsDistance];

						input[inputIndex + l * inputMiniBatchDistance] += currentWeight * output[outputIndex + l * outputMiniBatchDistance];

//						CustomArray ca = new CustomArray(new int[] {
//								inputIndex + l * inputMiniBatchDistance,
//								weightsStartIndex + k * weightsOutputFiltersDistance + fm * weightsInputFiltersDistance + weightsRow * weightsRowsDistance + weightsCol * weightsColumnsDistance,
//								outputIndex + l * outputMiniBatchDistance		
//						});
//
//						if (!CustomArrays.getInstance().getCustomArrays().get(c).contains(ca)) {
//							throw new RuntimeException(Arrays.toString(ca.getArray()));
//						} else {
//							CustomArrays.getInstance().getCustomArrays().get(c).remove(ca);
//						}
					}
				}
			}
		}
	}

	public boolean accept(Conv2DConnection c, ValuesProvider valuesProvider)
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

	public int getMiniBatchSize()
	{
		return miniBatchSize;
	}

	public int getFilterRows()
	{
		return filterRows;
	}

	public int getFilterCols()
	{
		return filterCols;
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

	public int getOutputFeatureMaps()
	{
		return outputFeatureMaps;
	}

	public int getWeightsRowsDistance()
	{
		return weightsRowsDistance;
	}

	public int getWeightsColumnsDistance()
	{
		return weightsColumnsDistance;
	}

	public AparapiBackpropagationConv2DWeightUpdates getAparapiWeightUpdates()
	{
		return aparapiWeightUpdates;
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

	public int getClear()
	{
		return clear;
	}
}
