package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import java.util.HashMap;
import java.util.Map;

import com.github.neuralnetworks.calculation.operations.aparapi.bp.AparapiMSELossFunction.MSE;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.AparapiMSELossFunction.MSEDerivative;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLLossFunction;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.util.Environment;

public class OpenCLMSE extends OpenCLLossFunction
{
	private static final long serialVersionUID = 1L;

	private transient MSE mse;

	@Override
	public float getLossFunction(Tensor activation, Tensor target)
	{
		if (mse == null)
		{
			mse = new MSE(activation, target, TensorFactory.tensor(activation.getDimensions()[0]));
		}

		Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(mse, activation.getDimensions()[0]);

		float result = 0;
		for (int i = 0; i < mse.getResult().length; i++)
		{
			result += mse.getResult()[i];
		}

		return result;
	}

	@Override
	protected OpenCLKernelReference createDerivative(OpenCLArrayReference activation, OpenCLArrayReference target, OpenCLArrayReference result)
	{
		int did = deviceId != null ? deviceId : OpenCLCore.getInstance().getAvailableDevices().get(0);

		MSEDerivative aparapi = new MSEDerivative(this.activationDerivative, this.targetDerivative, this.resultDerivative);
		int id = OpenCLCore.getInstance().MSEDerivative(did, activation.getId(), target.getId(), result.getId(), aparapi.getMiniBatchSize(), aparapi.getActivationStartPosition(), aparapi.getActivationRowStep(), aparapi.getActivationColumnStep(), aparapi.getTargetStartPosition(), aparapi.getTargetRowStep(), aparapi.getTargetColumnStep(), aparapi.getResultStartPosition(), aparapi.getResultRowStep(), aparapi.getResultColumnStep());

		return new OpenCLKernelReference(did, id, getModifiedArrays());
	}

	@Override
	protected OpenCLKernelReference createLossFunction(OpenCLArrayReference activation, OpenCLArrayReference target, OpenCLArrayReference result)
	{
		int did = deviceId != null ? deviceId : OpenCLCore.getInstance().getAvailableDevices().get(0);

		MSE aparapi = new MSE(this.activationLossFunction, this.targetLossFunction, this.resultLossFunction);
		int id = OpenCLCore.getInstance().MSE(did, activation.getId(), target.getId(), result.getId(), aparapi.getMiniBatchSize(), aparapi.getActivationStartPosition(), aparapi.getActivationRowStep(), aparapi.getActivationColumnStep(), aparapi.getTargetStartPosition(), aparapi.getTargetRowStep(), aparapi.getTargetColumnStep(), aparapi.getResultStartPosition(), aparapi.getResultRowStep(), aparapi.getResultColumnStep());

		return new OpenCLKernelReference(did, id, getModifiedArrays());
	}

	@Override
	public String kernelOptions(int order)
	{
		String result = null;

		if (isDerivative)
		{
			MSEDerivative aparapi = new MSEDerivative(this.activationDerivative, this.targetDerivative, this.resultDerivative);

			Map<String, String> fieldsMap = new HashMap<>();

			fieldsMap.put("activationStartPosition", "MSDaSP");
			fieldsMap.put("activationRowStep", "MSDaRS");
			fieldsMap.put("activationColumnStep", "MSDaCS");
			fieldsMap.put("targetStartPosition", "MSDtSP");
			fieldsMap.put("targetRowStep", "MSDtRS");
			fieldsMap.put("targetColumnStep", "MSDtCS");
			fieldsMap.put("resultStartPosition", "MSDrSP");
			fieldsMap.put("resultRowStep", "MSDrST");
			fieldsMap.put("resultColumnStep", "MSDrCS");

			result = OpenCLCore.getKernelOptionsString(aparapi, fieldsMap, order);
		} else if (isLossFunction)
		{
			MSE aparapi = new MSE(this.activationLossFunction, this.targetLossFunction, this.resultLossFunction);

			Map<String, String> fieldsMap = new HashMap<>();

			fieldsMap.put("activationStartPosition", "MSEaSP");
			fieldsMap.put("activationRowStep", "MSEaRS");
			fieldsMap.put("activationColumnStep", "MSEaCS");
			fieldsMap.put("targetStartPosition", "MSEtSP");
			fieldsMap.put("targetRowStep", "MSEtRS");
			fieldsMap.put("targetColumnStep", "MSEtCS");
			fieldsMap.put("resultStartPosition", "MSErSP");
			fieldsMap.put("resultRowStep", "MSErRS");
			fieldsMap.put("resultColumnStep", "MSErCS");

			result = OpenCLCore.getKernelOptionsString(aparapi, fieldsMap, order);
		}

		return result;
	}

	@Override
	public void destroyKernel()
	{
		super.destroyKernel();
		mse = null;
	}
}
