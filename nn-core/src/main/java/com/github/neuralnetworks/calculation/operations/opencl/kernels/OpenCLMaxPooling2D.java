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

public class OpenCLMaxPooling2D extends OpenCLConnectionCalculator
{
	private static final long serialVersionUID = 1L;

	private transient AparapiMaxPooling2DCC aparapi;

	public OpenCLMaxPooling2D()
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

		range = ((Subsampling2DConnection) connections.get(0)).getOutputLayer().getUnitCount(connections);

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

		int id = OpenCLCore.getInstance().MaxPooling2DCC(deviceId, inputRef.getId(), featureMapOffsetsId, outputRef.getId(), range, aparapi.getMiniBatchSize(), aparapi.getInputStartIndex(), aparapi.getInputMiniBatchDistance(), aparapi.getInputFeatureMapColumnsDistance(), aparapi.getInputFeatureMapRowsDistance(), aparapi.getInputFeatureMapsDistance(), aparapi.getOutputStartIndex(), aparapi.getOutputFeatureMapsDistance(), aparapi.getOutputFeatureMapLength(), aparapi.getOutputFeatureMapColumns(), aparapi.getOutputFeatureMapRowsDistance(), aparapi.getOutputFeatureMapColumnsDistance(), aparapi.getOutputMiniBatchDistance(), aparapi.getIoRowsOffset(), aparapi.getIoColumnsOffset(), aparapi.getRowStride(), aparapi.getColumnStride(), aparapi.getRegionLength());

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

		fieldsMap.put("miniBatchSize", "MpmBS");
		fieldsMap.put("inputStartIndex", "MpiSI");
		fieldsMap.put("inputMiniBatchDistance", "MpiMBD");
		fieldsMap.put("inputFeatureMapColumnsDistance", "MpiFCD");
		fieldsMap.put("inputFeatureMapRowsDistance", "MpiFRD");
		fieldsMap.put("inputFeatureMapsDistance", "MpiFMD");
		fieldsMap.put("outputStartIndex", "MpoSI");
		fieldsMap.put("outputFeatureMapsDistance", "MpoFMD");
		fieldsMap.put("outputFeatureMapLength", "MpoFML");
		fieldsMap.put("outputFeatureMapColumns", "MpoFMC");
		fieldsMap.put("outputFeatureMapRowsDistance", "MpoFRD");
		fieldsMap.put("outputFeatureMapColumnsDistance", "MpoFCD");
		fieldsMap.put("outputMiniBatchDistance", "MpoMBD");
		fieldsMap.put("ioRowsOffset", "MpioRO");
		fieldsMap.put("ioColumnsOffset", "MpioCO");
		fieldsMap.put("rowStride", "MprS");
		fieldsMap.put("columnStride", "MpcS");
		fieldsMap.put("regionLength", "MprL");

		Map<String, Object> kernelOptions = OpenCLCore.getKernelOptions(aparapi, fieldsMap);
		kernelOptions.put("MpiSX", aparapi.getInputStartIndex() + aparapi.getIoRowsOffset() * aparapi.getInputFeatureMapRowsDistance() + aparapi.getIoColumnsOffset() * aparapi.getInputFeatureMapColumnsDistance());
		kernelOptions.put("MpiFCDcS", aparapi.getInputFeatureMapColumnsDistance() * aparapi.getColumnStride());
		kernelOptions.put("MpiFRDrS", aparapi.getInputFeatureMapRowsDistance() * aparapi.getRowStride());
		kernelOptions.put("MpNDBS", range);

		StringBuilder result = new StringBuilder();
		kernelOptions.entrySet().forEach(e -> result.append(" -D ").append(e.getKey()).append(order).append("=").append(e.getValue()));

		return result.toString();
	}
}
