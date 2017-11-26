package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiMaxPooling2D.AparapiMaxPooling2DCC;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.util.Environment;

public class OpenCLStochasticPooling2D extends OpenCLConnectionCalculator
{
	private static final long serialVersionUID = 1L;

	private transient AparapiMaxPooling2DCC aparapi;

	public OpenCLStochasticPooling2D()
	{
		super();
	}

	@Override
	public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		if (aparapi == null)
		{
			aparapi = new AparapiMaxPooling2DCC((Subsampling2DConnection) connections.get(0), valuesProvider, targetLayer);
		}

		super.calculate(connections, valuesProvider, targetLayer);
	}

	@Override
	public OpenCLKernelReference createKernel()
	{
		this.deviceId = this.deviceId != null ? this.deviceId : Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getPreferredDevice();

		OpenCLArrayReferenceManager rm = OpenCLArrayReferenceManager.getInstance();

		OpenCLArrayReference inputRef = rm.getArrayReference(aparapi.getInput(), deviceId);
		if (inputRef == null) {
			rm.addToDevice(deviceId, aparapi.getInput(), 0);
			inputRef = rm.getArrayReference(aparapi.getInput(), deviceId);
		}

		OpenCLArrayReference outputRef = rm.getArrayReference(aparapi.getOutput(), deviceId);
		if (outputRef == null) {
			rm.addToDevice(deviceId, aparapi.getOutput(), 0);
			outputRef = rm.getArrayReference(aparapi.getOutput(), deviceId);
		}

		int featureMapOffsetsId = OpenCLCore.getInstance().prepareIntConstArray(deviceId, aparapi.getFeatureMapOffsets(), 0);

		int id = OpenCLCore.getInstance().StochasticPooling2DCC(deviceId, inputRef.getId(), featureMapOffsetsId, outputRef.getId(), range, aparapi.getMiniBatchSize(), aparapi.getInputStartIndex(), aparapi.getInputMiniBatchDistance(), aparapi.getInputFeatureMapColumnsDistance(), aparapi.getInputFeatureMapRowsDistance(), aparapi.getInputFeatureMapsDistance(), aparapi.getOutputStartIndex(), aparapi.getOutputFeatureMapsDistance(), aparapi.getOutputFeatureMapLength(), aparapi.getOutputFeatureMapColumns(), aparapi.getOutputFeatureMapRowsDistance(), aparapi.getOutputFeatureMapColumnsDistance(), aparapi.getOutputMiniBatchDistance(), aparapi.getIoRowsOffset(), aparapi.getIoColumnsOffset(), aparapi.getRowStride(), aparapi.getColumnStride(), aparapi.getRegionLength());

		return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
	}

	@Override
	public void destroyKernel()
	{
		super.destroyKernel();
		aparapi = null;
	}

	@Override
	public String kernelOptions(int order)
	{
		Map<String, String> fieldsMap = new HashMap<>();

		fieldsMap.put("miniBatchSize", "SpmBS");
		fieldsMap.put("inputStartIndex", "SpiSI");
		fieldsMap.put("inputMiniBatchDistance", "SpiMBD");
		fieldsMap.put("inputFeatureMapColumnsDistance", "SpiFCD");
		fieldsMap.put("inputFeatureMapRowsDistance", "SpiFRD");
		fieldsMap.put("inputFeatureMapsDistance", "SpiFMD");
		fieldsMap.put("outputStartIndex", "SpoSI");
		fieldsMap.put("outputFeatureMapsDistance", "SpoFMD");
		fieldsMap.put("outputFeatureMapLength", "SpoFML");
		fieldsMap.put("outputFeatureMapColumns", "SpoFMC");
		fieldsMap.put("outputFeatureMapRowsDistance", "SpoFRD");
		fieldsMap.put("outputFeatureMapColumnsDistance", "SpoFCD");
		fieldsMap.put("outputMiniBatchDistance", "SpoMBD");
		fieldsMap.put("ioRowsOffset", "SpioRO");
		fieldsMap.put("ioColumnsOffset", "SpioCO");
		fieldsMap.put("rowStride", "SprS");
		fieldsMap.put("columnStride", "SpcS");
		fieldsMap.put("regionLength", "SprL");

		return OpenCLCore.getKernelOptionsString(aparapi, fieldsMap, order);
	}
}
