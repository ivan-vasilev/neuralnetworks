package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiConv2DFF;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.util.Environment;

public class OpenCLConv2DFF extends OpenCLConnectionCalculator
{
	private static final long serialVersionUID = 1L;

	private transient AparapiConv2DFF aparapi;
	private Tensor weights;

	@Override
	public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		if (aparapi == null)
		{
			aparapi = new AparapiConv2DFF((Conv2DConnection) connections.get(0), valuesProvider, targetLayer);
		}

		if (weights == null)
		{
			weights = ((Conv2DConnection) connections.get(0)).getWeights();
		}

		range = ((Conv2DConnection) connections.get(0)).getOutputLayer().getUnitCount(connections);

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

		OpenCLArrayReference weightsRef = rm.getArrayReference(aparapi.getWeights(), deviceId);
		if (weightsRef == null) {
			rm.addToDevice(deviceId, aparapi.getWeights(), 0);
			weightsRef = rm.getArrayReference(aparapi.getWeights(), deviceId);
		}

		OpenCLArrayReference outputRef = rm.getArrayReference(aparapi.getOutput(), deviceId);
		if (outputRef == null) {
			rm.addToDevice(deviceId, aparapi.getOutput(), 0);
			outputRef = rm.getArrayReference(aparapi.getOutput(), deviceId);
		}

		boolean clear = aparapi.getClear() == 0 ? true : false;

		int featureMapOffsetsId = OpenCLCore.getInstance().prepareIntConstArray(deviceId, aparapi.getFeatureMapOffsets(), 0);

		int id = OpenCLCore.getInstance().Conv2DFF(deviceId, inputRef.getId(), weightsRef.getId(), outputRef.getId(), featureMapOffsetsId, range, aparapi.getInputStartIndex(), aparapi.getInputFeatureMapRowsDistance(), aparapi.getInputFeatureMapColumnsDistance(), aparapi.getFeatureMapWeights(), aparapi.getOutputColumns(), aparapi.getOutputStartIndex(), aparapi.getOutputFeatureMapLength(), aparapi.getOutputFeatureMapsDistance(), aparapi.getOutputFeatureMapColumnsDistance(), aparapi.getOutputFeatureMapRowsDistance(), aparapi.getWeightsStartIndex(), aparapi.getMiniBatchSize(), aparapi.getInputMiniBatchDistance(), aparapi.getOutputMiniBatchDistance(), weights.getDimensions()[1], aparapi.getRowStride(), aparapi.getColumnStride(), weights.getDimensions()[2], weights.getDimensions()[3], clear);

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

		fieldsMap.put("inputStartIndex", "C2iSI");
		fieldsMap.put("featureMapWeights", "C2fMW");
		fieldsMap.put("outputColumns", "C2oC");
		fieldsMap.put("outputStartIndex", "C2oSI");
		fieldsMap.put("outputFeatureMapLength", "C2oFML");
		fieldsMap.put("outputFeatureMapsDistance", "C2oFMD");
		fieldsMap.put("outputFeatureMapColumnsDistance", "C2oFMCD");
		fieldsMap.put("outputFeatureMapRowsDistance", "C2oFMRD");
		fieldsMap.put("weightsStartIndex", "C2wSI");
		fieldsMap.put("miniBatchSize", "C2mBS");
		fieldsMap.put("inputMiniBatchDistance", "C2iMBD");
		fieldsMap.put("outputMiniBatchDistance", "C2oMBD");
		fieldsMap.put("getFilterColumnsm1", "C2fCm1");
		fieldsMap.put("getFilterRowsm1", "C2fRm1");

		Map<String, Object> kernelOptions = OpenCLCore.getKernelOptions(aparapi, fieldsMap);
		kernelOptions.put("C2iND", range);
		kernelOptions.put("C2iFRDxRS", aparapi.getInputFeatureMapRowsDistance() * aparapi.getRowStride());
		kernelOptions.put("C2iFCDxCS", aparapi.getInputFeatureMapColumnsDistance() * aparapi.getColumnStride());

		int getFilterColumnsm1 = aparapi.getConnection().getFilterColumns() > 0 ? aparapi.getConnection().getFilterColumns() - 1 : 0;
		int getFilterRowsm1 = aparapi.getConnection().getFilterRows() > 0 ? aparapi.getConnection().getFilterRows() - 1 : 0;
		kernelOptions.put("C2fCm1", getFilterColumnsm1);
		kernelOptions.put("C2fRm1", getFilterRowsm1);
		kernelOptions.put("C2iFRDkmax", aparapi.getInputFeatureMapRowsDistance() - getFilterColumnsm1);
		kernelOptions.put("C2iMBDjkmax", input.getDimensionElementsDistance(1) - getFilterColumnsm1 * aparapi.getInputFeatureMapColumnsDistance() - getFilterRowsm1 * aparapi.getInputFeatureMapRowsDistance());

		StringBuilder result = new StringBuilder();
		kernelOptions.entrySet().forEach(e -> result.append(" -D ").append(e.getKey()).append(order).append("=").append(e.getValue()));

		return result.toString();
	}
}
