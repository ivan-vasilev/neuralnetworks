package com.github.neuralnetworks.calculation.operations.opencl;

import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.training.backpropagation.LossFunction;
import com.github.neuralnetworks.util.Environment;

public abstract class OpenCLLossFunction implements LossFunction, OpenCLKernelData
{
	private static final long serialVersionUID = 1L;

	protected transient Integer deviceId;
	protected Tensor activationDerivative;
	protected Tensor resultDerivative;
	protected Tensor targetDerivative;
	protected Tensor activationLossFunction;
	protected Tensor resultLossFunction;
	protected Tensor targetLossFunction;
	protected int range;

	protected boolean isDerivative;
	protected boolean isLossFunction;

	private OpenCLKernelReference lossFunctionReference;
	private OpenCLKernelReference derivativeReference;

	@Override
	public Tensor getInput()
	{
		if (isDerivative)
		{
			return activationDerivative;
		}

		if (isLossFunction)
		{
			return activationLossFunction;
		}

		return null;
	}

	@Override
	public Tensor getOutput()
	{
		if (isDerivative)
		{
			return resultDerivative;
		}

		if (isLossFunction)
		{
			return resultLossFunction;
		}

		return null;
	}

	@Override
	public OpenCLKernelReference createKernel()
	{
		OpenCLKernelReference reference = null;
		OpenCLArrayReferenceManager rm = OpenCLArrayReferenceManager.getInstance();
		this.deviceId = this.deviceId != null ? this.deviceId : Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getPreferredDevice();

		if (isDerivative)
		{
			rm.addToDevice(deviceId, activationDerivative);
			rm.addToDevice(deviceId, targetDerivative);
			rm.addToDevice(deviceId, resultDerivative);

			range = activationDerivative.getDimensions()[0];

			reference = createDerivative(rm.getArrayReference(activationDerivative, deviceId), rm.getArrayReference(targetDerivative, deviceId), rm.getArrayReference(resultDerivative, deviceId));
		} else if (isLossFunction)
		{
			if (resultLossFunction == null)
			{
				resultLossFunction = TensorFactory.tensor(activationLossFunction.getDimensions()[0]);
			}

			rm.addToDevice(deviceId, activationLossFunction);
			rm.addToDevice(deviceId, targetLossFunction);
			rm.addToDevice(deviceId, resultLossFunction);

			range = activationLossFunction.getDimensions()[0];

			reference = createLossFunction(rm.getArrayReference(activationLossFunction, deviceId), rm.getArrayReference(targetLossFunction, deviceId), rm.getArrayReference(resultLossFunction, deviceId));
		}

		return reference;
	}

	@Override
	public void destroyKernel()
	{ 
		activationDerivative = null;
		resultDerivative = null;
		targetDerivative = null;
		activationLossFunction = null;
		resultLossFunction = null;
		targetLossFunction = null;
		lossFunctionReference = null;
		derivativeReference = null;
		range = 0;

		isDerivative = false;
		isLossFunction = false;
	}

	protected abstract OpenCLKernelReference createDerivative(OpenCLArrayReference activation, OpenCLArrayReference target, OpenCLArrayReference result);

	protected abstract OpenCLKernelReference createLossFunction(OpenCLArrayReference activation, OpenCLArrayReference target, OpenCLArrayReference result);

	@Override
	public float getLossFunction(Tensor activation, Tensor target)
	{
		setLossFunctionMode();

		if (this.activationLossFunction != activation)
		{
			this.activationLossFunction = activation;
		}

		if (this.targetLossFunction != target)
		{
			this.targetLossFunction = target;
		}

//		if (lossFunctionReference == null || lossFunctionReference.getId() == null) 
//		{
//			lossFunctionReference = createKernel();
//		}

		if (OpenCLKernelReferenceManager.getInstance().get(this) != lossFunctionReference)
		{
			OpenCLKernelReferenceManager.getInstance().put(this, lossFunctionReference);
		}

		OpenCLKernelsExecutor.getInstance().execute(this);

		OpenCLArrayReferenceManager.getInstance().pushToDevice(resultLossFunction.getElements());

		float loss = 0;
		for (int i = 0; i < resultLossFunction.getElements().length; i++)
		{
			loss += resultLossFunction.getElements()[i];
		}

		return loss;
	}

	@Override
	public void getLossFunctionDerivative(Tensor activation, Tensor target, Tensor result)
	{
		setDerivativeMode();

		if (this.activationDerivative != activation)
		{
			this.activationDerivative = activation;
		}

		if (this.targetDerivative != target)
		{
			this.targetDerivative = target;
		}

		if (this.resultDerivative != result)
		{
			this.resultDerivative = result;
		}

//		if (derivativeReference == null || derivativeReference.getId() == null) 
//		{
//			derivativeReference = createKernel();
//		}

		if (OpenCLKernelReferenceManager.getInstance().get(this) != derivativeReference)
		{
			OpenCLKernelReferenceManager.getInstance().put(this, derivativeReference);
		}

		OpenCLKernelsExecutor.getInstance().execute(this);
	}

	public void setLossFunctionMode() 
	{
		isLossFunction = true;
		isDerivative = false;
	}
	
	public void setDerivativeMode() 
	{
		isDerivative = true;
		isLossFunction = false;
	}

	public boolean isDerivative()
	{
		return isDerivative;
	}

	public boolean isLossFunction()
	{
		return isLossFunction;
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
