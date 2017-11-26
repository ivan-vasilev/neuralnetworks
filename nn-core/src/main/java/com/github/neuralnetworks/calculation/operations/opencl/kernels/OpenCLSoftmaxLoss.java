package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import java.util.HashMap;
import java.util.Map;

import com.github.neuralnetworks.calculation.operations.aparapi.bp.AparapiSoftmaxLossFunction.NegativeLogProbability;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.AparapiSoftmaxLossFunction.SoftmaxLoss;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLLossFunction;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.util.Environment;

public class OpenCLSoftmaxLoss extends OpenCLLossFunction
{
	private static final long serialVersionUID = 1L;

	private transient NegativeLogProbability nlp;

	@Override
	public float getLossFunction(Tensor activation, Tensor target)
	{
		if (nlp == null)
		{
			nlp = new NegativeLogProbability(activation, target, TensorFactory.tensor(activation.getDimensions()[0]));
		}

		Environment.getInstance().getRuntimeConfiguration().getAparapiConfiguration().getExecutionStrategy().execute(nlp, activation.getDimensions()[0]);

		float result = 0;
		for (int i = 0; i < nlp.getResult().length; i++)
		{
			result += nlp.getResult()[i];
		}

		return result;
	}

	@Override
	protected OpenCLKernelReference createDerivative(OpenCLArrayReference activation, OpenCLArrayReference target, OpenCLArrayReference result)
	{
		int did = deviceId != null ? deviceId : OpenCLCore.getInstance().getAvailableDevices().get(0);

		SoftmaxLoss aparapi = new SoftmaxLoss(this.activationDerivative, this.targetDerivative, this.resultDerivative);
		int id = OpenCLCore.getInstance().SoftmaxLoss(did, activation.getId(), target.getId(), result.getId(), aparapi.getMiniBatchSize(), aparapi.getActivationStartPosition(), aparapi.getActivationRowStep(), aparapi.getActivationColumnStep(), aparapi.getTargetStartPosition(), aparapi.getTargetRowStep(), aparapi.getTargetColumnStep(), aparapi.getResultStartPosition(), aparapi.getResultRowStep(), aparapi.getResultColumnStep(), aparapi.getReverse());

		return new OpenCLKernelReference(did, id, getModifiedArrays());
	}

	@Override
	protected OpenCLKernelReference createLossFunction(OpenCLArrayReference activation, OpenCLArrayReference target, OpenCLArrayReference result)
	{
		int did = deviceId != null ? deviceId : OpenCLCore.getInstance().getAvailableDevices().get(0);

		SoftmaxLoss aparapi = new SoftmaxLoss(this.activationLossFunction, this.targetLossFunction, this.resultLossFunction);
		int id = OpenCLCore.getInstance().NegativeLogProbability(did, activation.getId(), target.getId(), result.getId(), aparapi.getMiniBatchSize(), this.activationLossFunction.getDimensions()[0], aparapi.getActivationStartPosition(), aparapi.getActivationRowStep(), aparapi.getActivationColumnStep(), aparapi.getTargetStartPosition(), aparapi.getTargetRowStep(), aparapi.getTargetColumnStep(), aparapi.getResultStartPosition(), aparapi.getResultRowStep(), aparapi.getResultColumnStep());

		return new OpenCLKernelReference(did, id, getModifiedArrays());
	}

	@Override
	public String kernelOptions(int order)
	{
		String result = null;

		if (isDerivative)
		{
			SoftmaxLoss aparapi = new SoftmaxLoss(this.activationDerivative, this.targetDerivative, this.resultDerivative);

			Map<String, String> fieldsMap = new HashMap<>();

			fieldsMap.put("activationStartPosition", "SMLaSP");
			fieldsMap.put("activationRowStep", "SMLaRS");
			fieldsMap.put("activationColumnStep", "SMLaCS");
			fieldsMap.put("targetStartPosition", "SMLtSP");
			fieldsMap.put("targetRowStep", "SMLtRS");
			fieldsMap.put("targetColumnStep", "SMLtCS");
			fieldsMap.put("resultStartPosition", "SMLrSP");
			fieldsMap.put("resultRowStep", "SMLrRS");
			fieldsMap.put("resultColumnStep", "SMLrCS");
			fieldsMap.put("reverse", "SMLrrv");

			result = OpenCLCore.getKernelOptionsString(aparapi, fieldsMap, order);
		} else if (isLossFunction)
		{
			NegativeLogProbability aparapi = new NegativeLogProbability(this.activationLossFunction, this.targetLossFunction, this.resultLossFunction);

			Map<String, String> fieldsMap = new HashMap<>();

			fieldsMap.put("miniBatchSize", "NLPmBS");
			fieldsMap.put("activationStartPosition", "NLPaSP");
			fieldsMap.put("activationRowStep", "NLPaRS");
			fieldsMap.put("activationColumnStep", "NLPaCS");
			fieldsMap.put("targetStartPosition", "NLPtSP");
			fieldsMap.put("targetRowStep", "NLPtRS");
			fieldsMap.put("targetColumnStep", "NLPtCS");
			fieldsMap.put("resultStartPosition", "NLPrSP");
			fieldsMap.put("resultRowStep", "NLPrRS");
			fieldsMap.put("resultColumnStep", "NLPrCS");

			result = OpenCLCore.getKernelOptionsString(aparapi, fieldsMap, order);
		}

		return result;
	}

	@Override
	public void destroyKernel()
	{
		super.destroyKernel();
		nlp = null;
	}
}
