package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.AparapiBackpropagationSubsampling2D2;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculatorImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;

public class OpenCLMaxPooling2DBP extends BackPropagationConnectionCalculatorImpl
{

	private static final long serialVersionUID = 1L;

	public OpenCLMaxPooling2DBP(Properties properties)
	{
		super(properties);
	}

	@Override
	protected void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider,
			ValuesProvider activations, Layer targetLayer)
	{
		Subsampling2DConnection con = null;
		for (Connections c : inputConnections)
		{
			if (c instanceof Subsampling2DConnection)
			{
				con = (Subsampling2DConnection) c;
				break;
			}
		}

		if (con != null)
		{
			connectionCalculators.put(con, new OpenCLMaxPooling2DBPCC());
		}
	}

	public static class OpenCLMaxPooling2DBPCC extends OpenCLConnectionCalculator implements BackPropagationConnectionCalculator
	{
		private static final long serialVersionUID = 1L;

		protected ValuesProvider activations;
		protected Tensor activationsTensor;
		protected transient AparapiBackpropagationSubsampling2D2 aparapi;

		@Override
		public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
		{
			if (connections.size() != 1)
			{
				throw new RuntimeException("Only one connection is allowed");
			}

			if (activationsTensor == null)
			{
				activationsTensor = TensorFactory.tensor(connections.get(0).getInputLayer(), connections.get(0), activations);
			}

			if (aparapi == null)
			{
				aparapi = new AparapiBackpropagationSubsampling2D2((Subsampling2DConnection) connections.get(0), valuesProvider, activations);
			}

			range = ((Subsampling2DConnection) connections.get(0)).getInputLayer().getUnitCount(connections);

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

			OpenCLArrayReference outputRef = rm.getArrayReference(aparapi.getOutput(), deviceId);
			if (outputRef == null)
			{
				rm.addToDevice(deviceId, aparapi.getOutput(), 0);
				outputRef = rm.getArrayReference(aparapi.getOutput(), deviceId);
			}

			OpenCLArrayReference activationsRef = rm.getArrayReference(activationsTensor, deviceId);
			if (activationsRef == null)
			{
				rm.addToDevice(deviceId, activationsTensor);
				activationsRef = rm.getArrayReference(activationsTensor, deviceId);
			}

			int featureMapOffsetsId = OpenCLCore.getInstance().prepareIntConstArray(deviceId, aparapi.getFeatureMapOffsets(), 0);

			int id = OpenCLCore.getInstance().BackpropagationMaxPooling2DCC(deviceId, inputRef.getId(), featureMapOffsetsId, activationsRef.getId(), outputRef.getId(), range, aparapi.getMiniBatchSize(), aparapi.getInputStartIndex(), aparapi.getInputMiniBatchDistance(), aparapi.getInputFeatureMapColumnsDistance(), aparapi.getInputFeatureMapRowsDistance(), aparapi.getInputFeatureMapsDistance(), aparapi.getInputFeatureMapLength(), aparapi.getInputFeatureMapColumns(), aparapi.getInputFeatureMapRows(), aparapi.getSubsamplingRows(), aparapi.getSubsamplingCols(), aparapi.getOutputMiniBatchDistance(), aparapi.getOutputStartIndex(), aparapi.getOutputFeatureMapsDistance(), aparapi.getOutputFeatureMapRowsDistance(), aparapi.getOutputFeatureMapColumnsDistance(), aparapi.getIoColumnsOffset(), aparapi.getIoRowsOffset(), aparapi.getRowStride(), aparapi.getColumnStride(), aparapi.getRegionLength());

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
			result.add(getInput().getElements());
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
			this.activations = activations;
		}

		@Override
		public String kernelOptions(int order)
		{
			Map<String, String> fieldsMap = new HashMap<>();

			fieldsMap.put("miniBatchSize", "BmmBS");
			fieldsMap.put("inputStartIndex", "BmiSI");
			fieldsMap.put("inputMiniBatchDistance", "BmiMBD");
			fieldsMap.put("inputFeatureMapColumnsDistance", "BmiFCD");
			fieldsMap.put("inputFeatureMapRowsDistance", "BmiFRD");
			fieldsMap.put("inputFeatureMapsDistance", "BmiFMD");
			fieldsMap.put("inputFeatureMapLength", "BmiFML");
			fieldsMap.put("inputFeatureMapColumns", "BmiFMC");
			fieldsMap.put("inputFeatureMapRows", "BmiFMR");
			fieldsMap.put("subsamplingRows", "BmsR");
			fieldsMap.put("subsamplingCols", "BmsC");
			fieldsMap.put("outputMiniBatchDistance", "BmoMBD");
			fieldsMap.put("outputStartIndex", "BmoSI");
			fieldsMap.put("outputFeatureMapsDistance", "BmoFMD");
			fieldsMap.put("outputFeatureMapRowsDistance", "BmoFRD");
			fieldsMap.put("outputFeatureMapColumnsDistance", "BmoFCD");
			fieldsMap.put("ioColumnsOffset", "BmioCO");
			fieldsMap.put("ioRowsOffset", "BmioRO");
			fieldsMap.put("rowStride", "BmrS");
			fieldsMap.put("columnStride", "BmcS");
			fieldsMap.put("regionLength", "BmrL");

			Map<String, Object> kernelOptions = OpenCLCore.getKernelOptions(aparapi, fieldsMap);
			kernelOptions.put("BmiFCDcS", aparapi.getColumnStride() * aparapi.getInputFeatureMapColumnsDistance());
			kernelOptions.put("BmiFRDrS", aparapi.getRowStride() * aparapi.getInputFeatureMapRowsDistance());
			kernelOptions.put("BmiFsR1", (aparapi.getInputFeatureMapRows() - aparapi.getSubsamplingRows()) / aparapi.getRowStride());
			kernelOptions.put("BmiFsC1", (aparapi.getInputFeatureMapColumns() - aparapi.getSubsamplingCols()) / aparapi.getColumnStride());
			kernelOptions.put("BmsR1", aparapi.getSubsamplingRows() - 1);
			kernelOptions.put("BmsC1", aparapi.getSubsamplingCols() - 1);
			kernelOptions.put("BmoSIN", aparapi.getOutputStartIndex() + aparapi.getIoRowsOffset() * aparapi.getOutputFeatureMapRowsDistance() + aparapi.getIoColumnsOffset() * aparapi.getOutputFeatureMapColumnsDistance());

			StringBuilder result = new StringBuilder();
			kernelOptions.entrySet().forEach(e -> result.append(" -D ").append(e.getKey()).append(order).append("=").append(e.getValue()));

			return result.toString();
		}
	}
}
