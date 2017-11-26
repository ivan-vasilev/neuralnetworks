package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.AparapiBackpropagationConv2DWeightUpdates;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelData;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelsExecutor;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.backpropagation.WeightUpdates;
import com.github.neuralnetworks.util.Environment;

public class OpenCLConv2DBPWeightUpdates implements OpenCLKernelData, TensorFunction, WeightUpdates
{
	private static final long serialVersionUID = 1L;

	protected transient AparapiBackpropagationConv2DWeightUpdates aparapi;
	protected Conv2DConnection connection;
	protected ValuesProvider valuesProvider;
	protected ValuesProvider activations;
	protected Tensor weightUpdates;
	protected transient Integer deviceId;

	public OpenCLConv2DBPWeightUpdates(Conv2DConnection connection, ValuesProvider valuesProvider, ValuesProvider activations, Tensor weightUpdates)
	{
		this.connection = connection;
		this.valuesProvider = valuesProvider;
		this.activations = activations;
		this.weightUpdates = weightUpdates;
	}

	@Override
	public void value(Tensor inputOutput)
	{
		OpenCLKernelsExecutor.getInstance().execute(this);
	}

	@Override
	public Tensor getInput()
	{
		return aparapi.getOutputTensor();
	}

	@Override
	public Tensor getOutput()
	{
		return connection.getWeights();
	}

	@Override
	public void updateWeights(float learningRate, float momentum, float l1weightDecay, float l2weightDecay)
	{
		if (aparapi == null)
		{
			aparapi = new AparapiBackpropagationConv2DWeightUpdates(connection, valuesProvider, activations, weightUpdates);
		}

		aparapi.setLearningRate(learningRate);
		aparapi.setMomentum(momentum);
		aparapi.setL1weightDecay(l1weightDecay);
		aparapi.setL2weightDecay(l2weightDecay);

		OpenCLKernelsExecutor.getInstance().execute(this);
	}

	@Override
	public OpenCLKernelReference createKernel()
	{
		this.deviceId = this.deviceId != null ? this.deviceId : Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getPreferredDevice();

		OpenCLArrayReferenceManager rm = OpenCLArrayReferenceManager.getInstance();

		OpenCLArrayReference weightsRef = rm.getArrayReference(aparapi.getWeights(), deviceId);
		if (weightsRef == null)
		{
			rm.addToDevice(deviceId, aparapi.getWeights(), 0);
			weightsRef = rm.getArrayReference(aparapi.getWeights(), deviceId);
		}
		
		OpenCLArrayReference weightsUpdatesRef = rm.getArrayReference(aparapi.getWeightsUpdates(), deviceId);
		if (weightsUpdatesRef == null)
		{
			rm.addToDevice(deviceId, aparapi.getWeightsUpdates(), 0);
			weightsUpdatesRef = rm.getArrayReference(aparapi.getWeightsUpdates(), deviceId);
		}

		OpenCLArrayReference outputRef = rm.getArrayReference(aparapi.getOutput(), deviceId);
		if (outputRef == null)
		{
			rm.addToDevice(deviceId, aparapi.getOutput(), 0);
			outputRef = rm.getArrayReference(aparapi.getOutput(), deviceId);
		}

		OpenCLArrayReference activationsRef = rm.getArrayReference(aparapi.getActivation(), deviceId);
		if (activationsRef == null)
		{
			rm.addToDevice(deviceId, aparapi.getActivation(), 0);
			activationsRef = rm.getArrayReference(aparapi.getActivation(), deviceId);
		}

		int id = OpenCLCore.getInstance().BackpropagationConv2DWeightUpdates(deviceId, activationsRef.getId(), weightsRef.getId(), weightsUpdatesRef.getId(), outputRef.getId(), connection.getWeights().getSize(), aparapi.getMiniBatchSize(), aparapi.getWeightsRowsDistance(), aparapi.getWeightsColumnsDistance(), aparapi.getWeightsStartIndex(), aparapi.getWeightsUpdatesOutputFiltersDistance(), aparapi.getWeightsUpdatesInputFiltersDistance(), aparapi.getWeightsInputFiltersDistance(), aparapi.getWeightsOutputFiltersDistance(), aparapi.getWeightsUpdatesRowsDistance(), aparapi.getActivationStartIndex(), aparapi.getActivationFeatureMapsDistance(), aparapi.getActivationFeatureMapRowsDistance(), aparapi.getActivationFeatureMapColumnsDistance(), aparapi.getActivationMiniBatchDistance(), aparapi.getOutputFeatureMapRows(), aparapi.getOutputFeatureMapColumns(), aparapi.getOutputMiniBatchDistance(), aparapi.getOutputStartIndex(), aparapi.getOutputFeatureMapsDistance(), aparapi.getOutputFeatureMapRowsDistance(), aparapi.getOutputFeatureMapColumnsDistance(), aparapi.getRowStride(), aparapi.getColumnStride(), aparapi.getLearningRate(), aparapi.getMomentum(), aparapi.getL1weightDecay(), aparapi.getL2weightDecay());

		return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
	}

	@Override
	public Set<float[]> getModifiedArrays() {
		Set<float[]> result = new HashSet<>();
		result.add(aparapi.getWeights());
		result.add(aparapi.getWeightsUpdates());
		return result;
	}

	@Override
	public void destroyKernel()
	{
		aparapi = null;
	}

	public AparapiBackpropagationConv2DWeightUpdates getAparapi()
	{
		return aparapi;
	}

	@Override
	public String kernelOptions(int order)
	{
		Map<String, String> fieldsMap = new HashMap<>();

		fieldsMap.put("miniBatchSize", "BWUmBS");
		fieldsMap.put("weightsRowsDistance", "BWUwRD");
		fieldsMap.put("weightsColumnsDistance", "BWUwCD");
		fieldsMap.put("weightsStartIndex", "BWUwSI");
		fieldsMap.put("weightsUpdatesOutputFiltersDistance", "BWUwuoFD");
		fieldsMap.put("weightsUpdatesInputFiltersDistance", "BWUwuiFD");
		fieldsMap.put("weightsInputFiltersDistance", "BWUwFD");
		fieldsMap.put("weightsOutputFiltersDistance", "BWUwoFD");
		fieldsMap.put("weightsUpdatesRowsDistance", "BWUwuRD");
		fieldsMap.put("activationStartIndex", "BWUaSI");
		fieldsMap.put("activationFeatureMapsDistance", "BWUaFMD");
		fieldsMap.put("activationFeatureMapRowsDistance", "BWUaFRD");
		fieldsMap.put("activationFeatureMapColumnsDistance", "BWUaCD");
		fieldsMap.put("activationMiniBatchDistance", "BWUaMBD");
		fieldsMap.put("outputFeatureMapRows", "BWUoFMR");
		fieldsMap.put("outputFeatureMapColumns", "BWUoFMC");
		fieldsMap.put("outputMiniBatchDistance", "BWUoMBD");
		fieldsMap.put("outputStartIndex", "BWUoSI");
		fieldsMap.put("outputFeatureMapsDistance", "BWUoFMD");
		fieldsMap.put("outputFeatureMapRowsDistance", "BWUoFRD");
		fieldsMap.put("outputFeatureMapColumnsDistance", "BWUoFCD");
		fieldsMap.put("rowStride", "BWUrS");
		fieldsMap.put("columnStride", "BWUcS");

		Map<String, Object> kernelOptions = OpenCLCore.getKernelOptions(aparapi, fieldsMap);

		kernelOptions.put("BWUlR", (aparapi.getLearningRate() / aparapi.getMiniBatchSize()) + "f");
		kernelOptions.put("BWUilR", (1 / (aparapi.getLearningRate() / aparapi.getMiniBatchSize())) + "f");

		kernelOptions.put("BWUm", aparapi.getMomentum() + "f");
		if (aparapi.getMomentum() > 0)
		{
			kernelOptions.put("BWUim", (1 / aparapi.getMomentum()) + "f");
		} else
		{
			kernelOptions.put("BWUim", "0");
		}

		kernelOptions.put("BWUl1", (-aparapi.getLearningRate() * aparapi.getL1weightDecay()) + "f");
		if (aparapi.getL1weightDecay() > 0)
		{
			kernelOptions.put("BWUil1", (1 / (-aparapi.getLearningRate() * aparapi.getL1weightDecay())) + "f");
		} else
		{
			kernelOptions.put("BWUil1", "0");
		}

		kernelOptions.put("BWUl2", (- aparapi.getL2weightDecay()) + "f");
		if (aparapi.getL2weightDecay() > 0)
		{
			kernelOptions.put("BWUil2", (1 / (-aparapi.getLearningRate() * aparapi.getL2weightDecay())) + "f");
		} else
		{
			kernelOptions.put("BWUil2", "0");
		}

		kernelOptions.put("BWUrSX", aparapi.getRowStride() * aparapi.getActivationFeatureMapRowsDistance());
		kernelOptions.put("BWUcSX", aparapi.getColumnStride() * aparapi.getActivationFeatureMapColumnsDistance());

		kernelOptions.put("BWUoFMRCM", aparapi.getMiniBatchSize() * aparapi.getOutputFeatureMapRows() * aparapi.getOutputFeatureMapColumns());

		kernelOptions.put("BWUrS", aparapi.getRowStride());
		kernelOptions.put("BWUcS", aparapi.getColumnStride());

		kernelOptions.put("BMUNDBS", connection.getWeights().getSize() * aparapi.getMiniBatchSize());

		StringBuilder result = new StringBuilder();
		kernelOptions.entrySet().forEach(e -> result.append(" -D ").append(e.getKey()).append(order).append("=").append(e.getValue()));

		return result.toString();
	}

	@Override
	public Integer getDeviceId()
	{
		return this.deviceId = this.deviceId != null ? this.deviceId : Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getPreferredDevice();
	}

	@Override
	public void setDeviceId(Integer deviceId)
	{
		this.deviceId = deviceId;
	}
}
