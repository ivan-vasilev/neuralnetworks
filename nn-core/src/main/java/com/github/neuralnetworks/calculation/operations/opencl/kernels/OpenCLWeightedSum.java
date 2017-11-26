package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.operations.aparapi.AparapiWeightedSum;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.util.Environment;

public class OpenCLWeightedSum extends OpenCLConnectionCalculator
{
	private static final long serialVersionUID = 1L;

	private transient AparapiWeightedSum aparapi;
	private Tensor weights;

	@Override
	public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		if (aparapi == null)
		{
			aparapi = new AparapiWeightedSum(connections.get(0), valuesProvider, targetLayer);
		}

		if (weights == null)
		{
			weights = ((FullyConnected) connections.get(0)).getWeights();
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

		int id = OpenCLCore.getInstance().weightedSum(deviceId, inputRef.getId(), weightsRef.getId(), outputRef.getId(), range, aparapi.getMiniBatchSize(), aparapi.getInputStartPosition(), aparapi.getInputRowStep(),
				aparapi.getInputColumnStep(), aparapi.getOutputStartPosition(), aparapi.getOutputRowStep(), aparapi.getOutputColumnStep(), aparapi.getWeightStartPosition(), aparapi.getWeightsSize(),
				aparapi.getWeightsInitialStep(), aparapi.getWeightsStep(), clear);

		return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
	}

	public AparapiWeightedSum getAparapi()
	{
		return aparapi;
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

		fieldsMap.put("miniBatchSize", "WSBS");
		fieldsMap.put("inputStartPosition", "WSiSP");
		fieldsMap.put("inputRowStep", "WSiRS");
		fieldsMap.put("inputColumnStep", "WSiCS");
		fieldsMap.put("outputStartPosition", "WSoSP");
		fieldsMap.put("outputRowStep", "WSoRS");
		fieldsMap.put("outputColumnStep", "WSoCS");
		fieldsMap.put("weightStartPosition", "WSwSP");
		fieldsMap.put("weightsSize", "WSwZ");
		fieldsMap.put("weightsInitialStep", "WSwIS");
		fieldsMap.put("weightsStep", "WSwS");
		fieldsMap.put("WSNidxmax", "WSwZ");
		fieldsMap.put("WSNjmax", "WSwIS");

		Map<String, Object> kernelOptions = OpenCLCore.getKernelOptions(aparapi, fieldsMap);
		kernelOptions.put("WSNDBS", range * aparapi.getMiniBatchSize());

		StringBuilder result = new StringBuilder();
		kernelOptions.entrySet().forEach(e -> result.append(" -D ").append(e.getKey()).append(order).append("=").append(e.getValue()));

		return result.toString();
	}
}
