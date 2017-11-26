package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.RepeaterConnection;
import com.github.neuralnetworks.calculation.operations.aparapi.bp.BackPropagationLRN.BPLRNKernel;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLConnectionCalculator;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.ValuesProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculatorImpl;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Properties;

public class OpenCLLRNBP extends BackPropagationConnectionCalculatorImpl
{
	private static final long serialVersionUID = 1L;

	private OpenCLLRN feedforward;

	public OpenCLLRNBP(Properties properties, OpenCLLRN feedforward)
	{
		super(properties);
		this.feedforward = feedforward;
	}

	@Override
	protected void addBackpropFunction(List<Connections> inputConnections, Map<Connections, BackPropagationConnectionCalculator> connectionCalculators, ValuesProvider valuesProvider,
			ValuesProvider activations, Layer targetLayer)
	{
		RepeaterConnection con = null;
		for (Connections c : inputConnections)
		{
			if (c instanceof RepeaterConnection)
			{
				con = (RepeaterConnection) c;
				break;
			}
		}

		if (con != null)
		{
			OpenCLLRNBPCC cc = new OpenCLLRNBPCC(feedforward, valuesProvider, activations, con);
			connectionCalculators.put(con, cc);
		}
	}

	public class OpenCLLRNBPCC extends OpenCLConnectionCalculator implements BackPropagationConnectionCalculator
	{
		private static final long serialVersionUID = 1L;
		
		protected transient BPLRNKernel aparapi;
		protected OpenCLLRN feedforward;
		protected RepeaterConnection connection;
		protected ValuesProvider valuesProvider;
		protected ValuesProvider activations;
		protected Tensor weightUpdates;

		public OpenCLLRNBPCC(OpenCLLRN feedforward, ValuesProvider valuesProvider, ValuesProvider activationsProvider, RepeaterConnection connection)
		{
			super();
			this.feedforward = feedforward;
			this.valuesProvider = valuesProvider;
			this.activations = activationsProvider;
			this.connection = connection;
		}

		@Override
		public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer)
		{
			if (connections.size() != 1)
			{
				throw new RuntimeException("Only one connection is allowed");
			}

			range = ((RepeaterConnection) connections.get(0)).getInputLayer().getUnitCount(connections);

			if (aparapi == null) 
			{
				aparapi = new BPLRNKernel(valuesProvider, activations, connection, targetLayer, feedforward.getAparapi().getCache(), feedforward.getAparapi().getN(), feedforward.getAparapi().getA(), feedforward.getAparapi().getB());
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
			
			OpenCLArrayReference cacheRef = rm.getArrayReference(feedforward.getAparapi().getCache(), deviceId);
			if (cacheRef == null) {
				rm.addToDevice(deviceId, aparapi.getCache(), 0);
				cacheRef = rm.getArrayReference(aparapi.getCache(), deviceId);
			}
			
			OpenCLArrayReference activationsRef = rm.getArrayReference(aparapi.getFFActivations(), deviceId);
			if (activationsRef == null) {
				rm.addToDevice(deviceId, aparapi.getFFActivations(), 0);
				activationsRef = rm.getArrayReference(aparapi.getFFActivations(), deviceId);
			}
			
			OpenCLArrayReference outputRef = rm.getArrayReference(aparapi.getOutput(), deviceId);
			if (outputRef == null) {
				rm.addToDevice(deviceId, aparapi.getOutput(), 0);
				outputRef = rm.getArrayReference(aparapi.getOutput(), deviceId);
			}
			
			int id = OpenCLCore.getInstance().BackPropagationLRN(deviceId, inputRef.getId(), cacheRef.getId(), activationsRef.getId(), outputRef.getId(), range, aparapi.getMiniBatchSize(), aparapi.getInputStartIndex(), aparapi.getInputMiniBatchDistance(), aparapi.getInputFeatureMapsLength(), aparapi.getInputFeatureMaps(), aparapi.getActivationsStartIndex(), aparapi.getActivationsFeatureMapsDistance(), aparapi.getActivationsFeatureMapsLength(), aparapi.getActivationsMiniBatchDistance(), aparapi.getOutputStartIndex(), aparapi.getInputFeatureMapsDistance(), aparapi.getN(), aparapi.getA(), aparapi.getB());
			
			return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
		}
		
		@Override
		public void destroyKernel()
		{
			super.destroyKernel();
			aparapi = null;
		}
		
		@Override
		public ValuesProvider getActivations()
		{
			return null;
		}
		
		@Override
		public void setActivations(ValuesProvider activations)
		{
		}

		@Override
		public String kernelOptions(int order)
		{
			Map<String, String> fieldsMap = new HashMap<>();

			fieldsMap.put("miniBatchSize", "BLRNmBS");
			fieldsMap.put("inputStartIndex", "BLRNiSI");
			fieldsMap.put("inputMiniBatchDistance", "BLRNMBD");
			fieldsMap.put("inputFeatureMapsLength", "BLRNiFML");
			fieldsMap.put("inputFeatureMaps", "BLRNiFM");
			fieldsMap.put("activationsStartIndex", "BLRNaSI");
			fieldsMap.put("activationsFeatureMapsDistance", "BLRNaFMD");
			fieldsMap.put("activationsFeatureMapsLength", "BLRNaFML");
			fieldsMap.put("activationsMiniBatchDistance", "BLRNaMBD");
			fieldsMap.put("outputStartIndex", "BLRNoSI");
			fieldsMap.put("inputFeatureMapsDistance", "BLRNiMD");
			fieldsMap.put("n", "BLRNn");
			fieldsMap.put("a", "BLRNa");
			fieldsMap.put("b", "BLRNb");

			return OpenCLCore.getKernelOptionsString(aparapi, fieldsMap, order);
		}
	}
}
