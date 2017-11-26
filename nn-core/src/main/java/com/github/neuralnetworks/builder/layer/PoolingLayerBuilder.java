package com.github.neuralnetworks.builder.layer;

import com.github.neuralnetworks.architecture.ConnectionFactory;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.activation.LayerUtil;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;
import com.github.neuralnetworks.builder.layer.structure.KernelUsageOptions;
import com.github.neuralnetworks.builder.layer.structure.MainFunctionsChangeable;
import com.github.neuralnetworks.builder.layer.structure.NamedSingleInputLayerBuilder;
import com.github.neuralnetworks.training.Hyperparameters;
import com.github.neuralnetworks.util.Constants;

/**
 * @author tmey
 */
public class PoolingLayerBuilder extends NamedSingleInputLayerBuilder implements MainFunctionsChangeable, KernelUsageOptions
{

	/**
	 * 0 means stride = filterSize
	 */
	private int strideRows = 0;
	private int strideColumns = 0;
	private int paddingRows = 0;
	private int paddingColumns = 0;
	private int poolingRows = 3;
	private int poolingColumns = 3;

	private ActivationType activationType = ActivationType.Nothing;
	private TransferFunctionType transferFunctionType = TransferFunctionType.Max_Polling2D;


	public PoolingLayerBuilder(int poolingSize)
	{
		super("PoolingLayer");

		this.setPoolingSize(poolingSize);
	}

	public PoolingLayerBuilder(int poolingRows, int poolingColumns)
	{
		super("PoolingLayer");

		this.setPoolingRows(poolingRows);
		this.setPoolingColumns(poolingColumns);
	}

	@Override
	protected Layer build(NeuralNetworkImpl neuralNetwork, String newLayerName, Layer inputLayer, Hyperparameters hyperparameters)
	{

		ConnectionFactory cf = neuralNetwork.getProperties().getParameter(Constants.CONNECTION_FACTORY);

		// initialize default parameter the right way

		int localStrideRows = strideRows;
		int localStrideColumns = strideColumns;

		if (localStrideColumns == 0)
		{
			localStrideColumns = poolingColumns;
		}

		if (localStrideRows == 0)
		{
			localStrideRows = poolingRows;
		}

		// search last connection

		int inputFMRows = -1;
		int inputFMCols = -1;
		int filters = -1;

		int[] layerDimension = inputLayer.getLayerDimension();
		if (layerDimension.length != 3)
		{
			throw new IllegalStateException("The current layer should be connected to a not 3 dimensional layer (" + layerDimension.length + ")");
		}

		inputFMRows = layerDimension[0];
		inputFMCols = layerDimension[1];
		filters = layerDimension[2];

		if (inputFMRows < 1 || inputFMCols < 1 || filters < 1)
		{
			throw new IllegalStateException("The inputLayer is");
		}

		// create new layer and the connection


		Layer newLayer = new Layer();

		Subsampling2DConnection subsampling2DConnection = cf.subsampling2D(inputLayer, newLayer, inputFMRows, inputFMCols, poolingRows, poolingColumns, filters, localStrideRows, localStrideColumns,
				paddingRows,
				paddingColumns);

		newLayer.setLayerDimension(new int[] { subsampling2DConnection.getOutputFeatureMapRowsWithPadding(), subsampling2DConnection.getOutputFeatureMapColumnsWithPadding(),
				subsampling2DConnection.getFilters() });
		newLayer.setName(newLayerName);

		if (neuralNetwork.getLayerCalculator() != null)
		{
			LayerUtil.changeActivationAndTransferFunction(neuralNetwork.getLayerCalculator(), newLayer, transferFunctionType, activationType);
		}

		// add new layer to the network
		neuralNetwork.addLayer(newLayer);

		return newLayer;
	}

	public void setPoolingSize(int size)
	{
		if (size <= 0)
		{
			throw new IllegalArgumentException("The filter size must be greater than 0!");
		}

		this.poolingColumns = size;
		this.poolingRows = size;
	}

	public void setPaddingSize(int size)
	{
		if (size < 0)
		{
			throw new IllegalArgumentException("The padding size must be equals or greater than 0!");
		}

		this.paddingColumns = size;
		this.paddingRows = size;
	}

	public void setStrideSize(int size)
	{
		if (size <= 0)
		{
			throw new IllegalArgumentException("The stride size must be greater than 0!");
		}

		this.strideColumns = size;
		this.strideRows = size;
	}

	public void setPoolingRows(int poolingRows)
	{
		if (poolingRows <= 0)
		{
			throw new IllegalArgumentException("The poolingRows must be greater than 0!");
		}
		this.poolingRows = poolingRows;
	}

	public void setPoolingColumns(int poolingColumns)
	{
		if (poolingColumns <= 0)
		{
			throw new IllegalArgumentException("The poolingColumns must be greater than 0!");
		}
		this.poolingColumns = poolingColumns;
	}

	public TransferFunctionType getTransferFunctionType()
	{
		return transferFunctionType;
	}

	public PoolingLayerBuilder setTransferFunctionType(TransferFunctionType transferFunctionType)
	{

		if (transferFunctionType == null)
		{
			throw new IllegalArgumentException("transferFunctionType must be not null!");
		}

		this.transferFunctionType = transferFunctionType;
		return this;
	}

	public ActivationType getActivationType()
	{
		return activationType;
	}

	public PoolingLayerBuilder setActivationType(ActivationType activationType)
	{

		if (activationType == null)
		{
			throw new IllegalArgumentException("activationType must be not null!");
		}

		this.activationType = activationType;
		return this;
	}

	public void setStrideRows(int strideRows)
	{
		if (strideRows <= 0)
		{
			throw new IllegalArgumentException("The strideRows must be greater than 0!");
		}
		this.strideRows = strideRows;
	}

	public void setStrideColumns(int strideColumns)
	{
		if (strideColumns <= 0)
		{
			throw new IllegalArgumentException("The strideColumns must be greater than 0!");
		}
		this.strideColumns = strideColumns;
	}

	public void setPaddingRows(int paddingRows)
	{
		if (paddingRows < 0)
		{
			throw new IllegalArgumentException("The paddingRows must be greater than 0!");
		}
		this.paddingRows = paddingRows;
	}

	public void setPaddingColumns(int paddingColumns)
	{
		if (paddingColumns < 0)
		{
			throw new IllegalArgumentException("The paddingColumns must be greater than 0!");
		}
		this.paddingColumns = paddingColumns;
	}

	@Override
	public String toString()
	{
		return "PoolingLayerBuilder{" +
				"name=" + (getName() != null ? getName() : DEFAULT_LAYER_NAME) +
				", inputLayer=" + getInputLayerName() +
				", inputLayer=" + getInputLayerName() +
				", strideRows=" + strideRows +
				", strideColumns=" + strideColumns +
				", paddingRows=" + paddingRows +
				", paddingColumns=" + paddingColumns +
				", poolingRows=" + poolingRows +
				", poolingColumns=" + poolingColumns +
				", activationType=" + activationType +
				", transferFunctionType=" + transferFunctionType +
				'}';
	}
}
