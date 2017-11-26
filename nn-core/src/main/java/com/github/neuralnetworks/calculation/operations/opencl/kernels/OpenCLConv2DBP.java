package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.AparapiBackpropagationConv2D2;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculatorImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;

/**
 * Convolutional Backpropagation connection calculator without activation function (for connections to the output layer)
 */
public class OpenCLConv2DBP extends BackPropagationConnectionCalculatorImpl
{
	private static final long serialVersionUID = 1178188233641224762L;

	public OpenCLConv2DBP(Properties properties)
	{
		super(properties);
	}

	@Override
	protected void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider,
			ValuesProvider activatinos, Layer targetLayer)
	{
		Conv2DConnection con = null;
		for (Connections c : inputConnections)
		{
			if (c instanceof Conv2DConnection)
			{
				con = (Conv2DConnection) c;
				break;
			}
		}

		if (con != null)
		{
			OpenCLConv2DBPCC cc = new OpenCLConv2DBPCC(con, valuesProvider, activations, getWeightUpdates().get(con));
			connectionCalculators.put(con, cc);
		}
	}

	public static class OpenCLConv2DBPCC extends OpenCLConnectionCalculator implements BackPropagationConnectionCalculator
	{
		private static final long serialVersionUID = 1L;

		protected transient AparapiBackpropagationConv2D2 aparapi;

		protected Conv2DConnection connection;
		protected ValuesProvider valuesProvider;
		protected ValuesProvider activations;
		protected Tensor weightUpdates;

		public OpenCLConv2DBPCC(Conv2DConnection connection, ValuesProvider valuesProvider, ValuesProvider activations, Tensor weightUpdates)
		{
			super();
			this.connection = connection;
			this.valuesProvider = valuesProvider;
			this.activations = activations;
			this.weightUpdates = weightUpdates;
		}

		@Override
		public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
		{
			if (connections.size() != 1)
			{
				throw new RuntimeException("Only one connection is allowed");
			}

			range = ((Conv2DConnection) connections.get(0)).getInputLayer().getUnitCount(connections);

			if (aparapi == null)
			{
				aparapi = new AparapiBackpropagationConv2D2(connection, valuesProvider, activations, weightUpdates);
			}

			super.calculate(connections, valuesProvider, targetLayer);
		}

		@Override
		public OpenCLKernelReference createKernel()
		{
			this.deviceId = this.deviceId != null ? this.deviceId : Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getPreferredDevice();

			OpenCLArrayReferenceManager rm = OpenCLArrayReferenceManager.getInstance();

			OpenCLArrayReference inputRef = rm.getArrayReference(aparapi.getInput(), deviceId);
			if (inputRef == null)
			{
				rm.addToDevice(deviceId, aparapi.getInput(), 0);
				inputRef = rm.getArrayReference(aparapi.getInput(), deviceId);
			}

			OpenCLArrayReference weightsRef = rm.getArrayReference(aparapi.getWeights(), deviceId);
			if (weightsRef == null)
			{
				rm.addToDevice(deviceId, aparapi.getWeights(), 0);
				weightsRef = rm.getArrayReference(aparapi.getWeights(), deviceId);
			}

			OpenCLArrayReference outputRef = rm.getArrayReference(aparapi.getOutput(), deviceId);
			if (outputRef == null)
			{
				rm.addToDevice(deviceId, aparapi.getOutput(), 0);
				outputRef = rm.getArrayReference(aparapi.getOutput(), deviceId);
			}

			boolean clear = aparapi.getClear() == 0 ? true : false;

			int id = OpenCLCore.getInstance().BackpropagationConv2D2(deviceId, inputRef.getId(), weightsRef.getId(), outputRef.getId(), range, aparapi.getMiniBatchSize(), aparapi.getInputStartIndex(), aparapi.getInputMiniBatchDistance(), aparapi.getInputFeatureMapLength(), aparapi.getInputFeatureMapColumns(), aparapi.getInputFeatureMapRows(), aparapi.getInputFeatureMapColumnsDistance(), aparapi.getInputFeatureMapRowsDistance(), aparapi.getInputFeatureMapsDistance(), aparapi.getFilterRows(), aparapi.getFilterCols(), aparapi.getOutputStartIndex(), aparapi.getOutputFeatureMapsDistance(), aparapi.getOutputFeatureMapRowsDistance(), aparapi.getOutputFeatureMapColumnsDistance(), aparapi.getOutputMiniBatchDistance(), aparapi.getOutputFeatureMaps(), aparapi.getIoRowsOffset(), aparapi.getIoColumnsOffset(), aparapi.getRowStride(), aparapi.getColumnStride(), aparapi.getWeightsInputFiltersDistance(), aparapi.getWeightsOutputFiltersDistance(), aparapi.getWeightsStartIndex(), aparapi.getWeightsRowsDistance(), aparapi.getWeightsColumnsDistance(), clear);

			return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
		}

		@Override
		public void destroyKernel()
		{
			super.destroyKernel();
			aparapi = null;
		}

		@Override
		public Set<float[]> getModifiedArrays()
		{
			Set<float[]> result = super.getModifiedArrays();
			result.add(aparapi.getInput());
			result.add(aparapi.getOutput());
			result.add(aparapi.getWeights());

			return result;
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

		@Override
		public String kernelOptions(int order)
		{
			Map<String, String> fieldsMap = new HashMap<>();

			fieldsMap.put("miniBatchSize", "BcmBS");
			fieldsMap.put("inputStartIndex", "BciSI");
			fieldsMap.put("inputMiniBatchDistance", "BciMBD");
			fieldsMap.put("inputFeatureMapLength", "BciFML");
			fieldsMap.put("inputFeatureMapColumns", "BciFMC");
			fieldsMap.put("inputFeatureMapRows", "BciFMR");
			fieldsMap.put("inputFeatureMapColumnsDistance", "BciFCD");
			fieldsMap.put("inputFeatureMapRowsDistance", "BciFRD");
			fieldsMap.put("inputFeatureMapsDistance", "BciFMD");
			fieldsMap.put("filterRows", "BcfR");
			fieldsMap.put("filterCols", "BcfC");
			fieldsMap.put("outputStartIndex", "BcoSI");
			fieldsMap.put("outputFeatureMapsDistance", "BcoFMD");
			fieldsMap.put("outputFeatureMapRowsDistance", "BcoFRD");
			fieldsMap.put("outputFeatureMapColumnsDistance", "BcoFCD");
			fieldsMap.put("outputMiniBatchDistance", "BcoMBD");
			fieldsMap.put("outputFeatureMaps", "BcoFM");
			fieldsMap.put("ioRowsOffset", "BcioRO");
			fieldsMap.put("ioColumnsOffset", "BcioCO");
			fieldsMap.put("rowStride", "BcrS");
			fieldsMap.put("columnStride", "BccS");
			fieldsMap.put("weightsInputFiltersDistance", "BcwIFD");
			fieldsMap.put("weightsOutputFiltersDistance", "BcwOFD");
			fieldsMap.put("weightsStartIndex", "BcwSI");
			fieldsMap.put("weightsRowsDistance", "BcwRD");
			fieldsMap.put("weightsColumnsDistance", "BcwCD");

			Map<String, Object> kernelOptions = OpenCLCore.getKernelOptions(aparapi, fieldsMap);
			kernelOptions.put("BcND", range);

			StringBuilder result = new StringBuilder();
			kernelOptions.entrySet().forEach(e -> result.append(" -D ").append(e.getKey()).append(order).append("=").append(e.getValue()));

			return result.toString();
		}
	}
}
