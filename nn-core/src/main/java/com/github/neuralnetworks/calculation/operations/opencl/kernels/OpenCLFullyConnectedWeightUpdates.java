package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.AparapiFullyConnectedWeightUpdates;
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

public class OpenCLFullyConnectedWeightUpdates implements OpenCLKernelData, TensorFunction, WeightUpdates
{
	private static final long serialVersionUID = 1L;

	protected transient AparapiFullyConnectedWeightUpdates aparapi;
	protected FullyConnected connection;
	protected ValuesProvider valuesProvider;
	protected ValuesProvider activations;
	protected Tensor weightUpdates;
	protected transient Integer deviceId;

	public OpenCLFullyConnectedWeightUpdates(FullyConnected connection, ValuesProvider valuesProvider, ValuesProvider activations, Tensor weightUpdates)
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
		return aparapi.getInputTensor();
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
			aparapi = new AparapiFullyConnectedWeightUpdates(connection, valuesProvider, activations, weightUpdates);
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
		
		OpenCLArrayReference weightsUpdatesRef = rm.getArrayReference(aparapi.getWeightUpdates(), deviceId);
		if (weightsUpdatesRef == null)
		{
			rm.addToDevice(deviceId, aparapi.getWeightUpdates(), 0);
			weightsUpdatesRef = rm.getArrayReference(aparapi.getWeightUpdates(), deviceId);
		}

		OpenCLArrayReference inputRef = rm.getArrayReference(aparapi.getInput(), deviceId);
		if (inputRef == null)
		{
			rm.addToDevice(deviceId, aparapi.getInput(), 0);
			inputRef = rm.getArrayReference(aparapi.getInput(), deviceId);
		}

		OpenCLArrayReference activationsRef = rm.getArrayReference(aparapi.getFfActivation(), deviceId);
		if (activationsRef == null)
		{
			rm.addToDevice(deviceId, aparapi.getFfActivation(), 0);
			activationsRef = rm.getArrayReference(aparapi.getFfActivation(), deviceId);
		}

		int id = OpenCLCore.getInstance().FullyConnectedWeightUpdates(deviceId, inputRef.getId(), weightsRef.getId(), weightsUpdatesRef.getId(), activationsRef.getId(), connection.getWeights().getSize(), aparapi.getMiniBatchSize(), aparapi.getInputStartPosition(), aparapi.getInputColumnStep(), aparapi.getInputRowStep(), aparapi.getActivationStartPosition(), aparapi.getActivationColumnStep(), aparapi.getActivationRowStep(), aparapi.getWeightStartPosition(), aparapi.getWeightsColumns(), aparapi.getWeightsRowsDistance(), aparapi.getWeightsColumnsDistance(), aparapi.getLearningRate(), aparapi.getMomentum(), aparapi.getL1weightDecay(), aparapi.getL2weightDecay()); 

		return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
	}

	@Override
	public Set<float[]> getModifiedArrays() {
		Set<float[]> result = new HashSet<>();
		result.add(aparapi.getWeights());
		result.add(aparapi.getWeightUpdates());
		return result;
	}

	@Override
	public void destroyKernel()
	{
		aparapi = null;
	}

	public AparapiFullyConnectedWeightUpdates getAparapi()
	{
		return aparapi;
	}

	@Override
	public String kernelOptions(int order)
	{
		Map<String, String> fieldsMap = new HashMap<>();

		fieldsMap.put("miniBatchSize", "FCWUmBS");
		fieldsMap.put("inputStartPosition", "FCWUiSP");
		fieldsMap.put("inputColumnStep", "FCWUiCS");
		fieldsMap.put("inputRowStep", "FCWUiRS");
		fieldsMap.put("activationStartPosition", "FCWUaSP");
		fieldsMap.put("activationColumnStep", "FCWUaCS");
		fieldsMap.put("activationRowStep", "FCWUaRS");
		fieldsMap.put("weightStartPosition", "FCWUwSP");
		fieldsMap.put("weightsColumns", "FCWUwC");
		fieldsMap.put("weightsRowsDistance", "FCWUwRD");
		fieldsMap.put("weightsColumnsDistance", "FCWUwCD");

		Map<String, Object> kernelOptions = OpenCLCore.getKernelOptions(aparapi, fieldsMap);
		kernelOptions.put("FCWUlR", (aparapi.getLearningRate() / aparapi.getMiniBatchSize()) + "f");
		kernelOptions.put("FCWUilR", (1 / (aparapi.getLearningRate() / aparapi.getMiniBatchSize())) + "f");

		kernelOptions.put("FCWUm", aparapi.getMomentum() + "f");
		if (aparapi.getMomentum() > 0)
		{
			kernelOptions.put("FCWUim", (1 / aparapi.getMomentum()) + "f");
			kernelOptions.put("FCWUi0m", "eeeEEE");
		} else
		{
			kernelOptions.put("FCWUim", "0");
			kernelOptions.put("FCWUi0m", "eee000");
		}

		kernelOptions.put("FCWUl1", (-aparapi.getLearningRate() * aparapi.getL1weightDecay()) + "f");
		if (aparapi.getL1weightDecay() > 0)
		{
			kernelOptions.put("FCWUil1", (1 / (-aparapi.getLearningRate() * aparapi.getL1weightDecay())) + "f");
			kernelOptions.put("FCWUil01", "eeeEEE");
		} else
		{
			kernelOptions.put("FCWUil1", "0");
			kernelOptions.put("FCWUil01", "eee000");
		}

		kernelOptions.put("FCWUl2", (- aparapi.getL2weightDecay()) + "f");
		if (aparapi.getL2weightDecay() > 0)
		{
			kernelOptions.put("FCWUil2", (1 / (-aparapi.getLearningRate() * aparapi.getL2weightDecay())) + "f");
			kernelOptions.put("FCWUil02", "eeeEEE");
		} else
		{
			kernelOptions.put("FCWUil2", "0");
			kernelOptions.put("FCWUil02", "eee000");
		}

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
