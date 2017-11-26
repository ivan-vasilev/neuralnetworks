package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.operations.aparapi.LRN.LRNKernel;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.util.Environment;

public class OpenCLLRN extends OpenCLConnectionCalculator
{
	private static final long serialVersionUID = 1L;

	private transient LRNKernel aparapi;

	private float k;
	private int n;
	private float a;
	private float b;

	public OpenCLLRN(float k, int n, float a, float b)
	{
		super();
		this.k = k;
		this.n = n;
		this.a = a;
		this.b = b;
	}

	@Override
	public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
	{
		super.calculate(connections, valuesProvider, targetLayer);

		if (aparapi == null)
		{
			aparapi = new LRNKernel(input, output, k, n, a, b);
		}
	}

	@Override
	public OpenCLKernelReference createKernel()
	{
		this.deviceId = this.deviceId != null ? this.deviceId : Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getPreferredDevice();

		if (aparapi == null)
		{
			aparapi = new LRNKernel(input, output, k, n, a, b);
		}

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

		OpenCLArrayReference cacheRef = rm.getArrayReference(aparapi.getCache(), deviceId);
		if (cacheRef == null) {
			rm.addToDevice(deviceId, aparapi.getCache(), 0);
			cacheRef = rm.getArrayReference(aparapi.getCache(), deviceId);
		}

		int id = OpenCLCore.getInstance().LRN(deviceId, inputRef.getId(), outputRef.getId(), cacheRef.getId(), range, aparapi.getInputFeatureMapsLength(), aparapi.getInputFeatureMaps(), aparapi.getInputStartIndex(), aparapi.getInputFeatureMapsDistance(), n, aparapi.getMiniBatchSize(), aparapi.getInputMiniBatchDistance(), aparapi.getOutputStartIndex(), k, a, b);

		return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
	}

	@Override
	public void destroyKernel()
	{
		super.destroyKernel();
		aparapi = null;
	}

	public LRNKernel getAparapi()
	{
		return aparapi;
	}

	@Override
	public Set<float[]> getModifiedArrays() {
		Set<float[]> result = super.getModifiedArrays();
		result.add(aparapi.getCache());
		return result;
	}

	@Override
	public String kernelOptions(int order)
	{
		Map<String, String> fieldsMap = new HashMap<>();

		fieldsMap.put("inputFeatureMapsLength", "LRFL");
		fieldsMap.put("inputFeatureMaps", "LRFM");
		fieldsMap.put("inputStartIndex", "LRSI");
		fieldsMap.put("inputFeatureMapsDistance", "LRFD");
		fieldsMap.put("n", "LRn");
		fieldsMap.put("miniBatchSize", "LRBS");
		fieldsMap.put("inputMiniBatchDistance", "LRBD");
		fieldsMap.put("outputStartIndex", "LRoS");
		fieldsMap.put("k", "LRk");
		fieldsMap.put("a", "LRa");
		fieldsMap.put("b", "LRb");

		return OpenCLCore.getKernelOptionsString(aparapi, fieldsMap, order);
	}
}
