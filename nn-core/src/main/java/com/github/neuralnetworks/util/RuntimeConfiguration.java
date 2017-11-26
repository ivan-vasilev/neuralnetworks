package com.github.neuralnetworks.util;

import java.io.Serializable;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.calculation.operations.aparapi.KernelExecutionStrategy;
import com.github.neuralnetworks.calculation.operations.aparapi.KernelExecutionStrategy.CPUKernelExecution;
import com.github.neuralnetworks.calculation.operations.aparapi.KernelExecutionStrategy.DefaultKernelExecution;
import com.github.neuralnetworks.calculation.operations.aparapi.KernelExecutionStrategy.GPUKernelExecution;
import com.github.neuralnetworks.calculation.operations.aparapi.KernelExecutionStrategy.JTPKernelExecution;
import com.github.neuralnetworks.calculation.operations.aparapi.KernelExecutionStrategy.MockExecution;
import com.github.neuralnetworks.calculation.operations.aparapi.KernelExecutionStrategy.SeqKernelExecution;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;

/**
 * Runtime configuration for the library
 */
public class RuntimeConfiguration implements Serializable
{

	private static final long serialVersionUID = 1L;

	/**
	 * The provider of the calculation
	 */
	public static enum CalculationProvider
	{
		APARAPI,
		OPENCL,
		CPU
	}

	private CalculationProvider calculationProvider;

	/**
	 * configuration for aparapi execution
	 */
	private AparapiConfiguration aparapiConfiguration;

	/**
	 * configuration for OpenCL execution
	 */
	private OpenCLConfiguration openCLConfiguration;

	/**
	 * Shared memory for calculations
	 */
	private boolean useDataSharedMemory;

	/**
	 * Shared memory for neural network connection weights
	 */
	private boolean useWeightsSharedMemory;

	/**
	 * reverse softmax to pass gradient check test
	 */
	private boolean reverseSoftmaxLoss;

	public RuntimeConfiguration()
	{
		super();
		aparapiConfiguration = new AparapiConfiguration();
		openCLConfiguration = new OpenCLConfiguration();
		useDataSharedMemory = false;
		useWeightsSharedMemory = false;
		calculationProvider = CalculationProvider.APARAPI;
	}

	public CalculationProvider getCalculationProvider()
	{
		return calculationProvider;
	}

	public void setCalculationProvider(CalculationProvider calculationProvider)
	{
		this.calculationProvider = calculationProvider;
	}

	public AparapiConfiguration getAparapiConfiguration()
	{
		return aparapiConfiguration;
	}

	public void setAparapiConfiguration(AparapiConfiguration aparapiConfiguration)
	{
		this.aparapiConfiguration = aparapiConfiguration;
	}

	public OpenCLConfiguration getOpenCLConfiguration()
	{
		return openCLConfiguration;
	}

	public void setOpenCLConfiguration(OpenCLConfiguration openCLConfiguration)
	{
		this.openCLConfiguration = openCLConfiguration;
	}

	public boolean getUseDataSharedMemory()
	{
		return useDataSharedMemory;
	}

	public void setUseDataSharedMemory(boolean useDataSharedMemory)
	{
		this.useDataSharedMemory = useDataSharedMemory;
	}

	public boolean getUseWeightsSharedMemory()
	{
		return useWeightsSharedMemory;
	}

	public void setUseWeightsSharedMemory(boolean useWeightsSharedMemory)
	{
		this.useWeightsSharedMemory = useWeightsSharedMemory;
	}

	public boolean getReverseSoftmaxLoss()
	{
		return reverseSoftmaxLoss;
	}

	public void setReverseSoftmaxLoss(boolean reverseSoftmaxLoss)
	{
		this.reverseSoftmaxLoss = reverseSoftmaxLoss;
	}

	public static class AparapiConfiguration implements Serializable
	{
		private static final long serialVersionUID = 1L;

		/**
		 * Determnines whether the code will be executed on the GPU or the CPU
		 */
		private KernelExecutionStrategy executionStrategy;

		/**
		 * whether to use skip execution
		 */
		private MockExecution mockExecution;

		public AparapiConfiguration()
		{
			super();
			this.executionStrategy = new DefaultKernelExecution();
		}

		public KernelExecutionStrategy getExecutionStrategy()
		{
			if (mockExecution != null)
			{
				return mockExecution;
			}

			return executionStrategy;
		}

		public void setExecutionMode(EXECUTION_MODE executionMode)
		{
			switch (executionMode) {
			case CPU:
				this.executionStrategy = new CPUKernelExecution();
				break;
			case SEQ:
				this.executionStrategy = new SeqKernelExecution();
				break;
			case JTP:
				this.executionStrategy = new JTPKernelExecution();
				break;
			case GPU:
				this.executionStrategy = new GPUKernelExecution();
				break;
			default:
				this.executionStrategy = new DefaultKernelExecution();
			}
		}

		public void setMockExecution(boolean mockExecution)
		{
			this.mockExecution = mockExecution ? new MockExecution() : null;
		}
	}

	public static class OpenCLConfiguration implements Serializable
	{
		private static final long serialVersionUID = 1L;

		/**
		 * aggregate the operations
		 */
		private boolean aggregateOperations;

		/**
		 * use options string
		 */
		private boolean useOptionsString;

		/**
		 * whether to push the results back to the host after each operation
		 */
		private boolean synchronizeAfterOpertation;

		/**
		 * whether to finalize the device after a phase
		 */
		private boolean finalyzeDeviceAfterPhase;

		/**
		 * whether to push the arrays to the device before each operation
		 */
		private boolean pushToDeviceBeforeOperation;

		/**
		 * restart library after phase
		 */
		private boolean restartLibraryAfterPhase;

		/**
		 * whether to push the arrays to the device before each operation
		 */
		private Integer preferredDevice;

		public OpenCLConfiguration()
		{
			super();
			this.aggregateOperations = true;
			this.finalyzeDeviceAfterPhase = true;
		}

		public boolean getAggregateOperations()
		{
			return aggregateOperations;
		}

		public void setAggregateOperations(boolean aggregateOperations)
		{
			this.aggregateOperations = aggregateOperations;
		}

		public boolean getSynchronizeAfterOpertation()
		{
			return synchronizeAfterOpertation;
		}

		public void setSynchronizeAfterOpertation(boolean synchronizeAfterOpertation)
		{
			this.synchronizeAfterOpertation = synchronizeAfterOpertation;
		}

		public boolean getPushToDeviceBeforeOperation()
		{
			return pushToDeviceBeforeOperation;
		}

		public void setPushToDeviceBeforeOperation(boolean pushToDeviceBeforeOperation)
		{
			this.pushToDeviceBeforeOperation = pushToDeviceBeforeOperation;
		}

		public boolean getFinalyzeDeviceAfterPhase()
		{
			return finalyzeDeviceAfterPhase;
		}

		public void setFinalyzeDeviceAfterPhase(boolean finalyzeDeviceAfterPhase)
		{
			this.finalyzeDeviceAfterPhase = finalyzeDeviceAfterPhase;
		}

		public int getPreferredDevice()
		{
			if (preferredDevice == null)
			{
				preferredDevice = OpenCLCore.getInstance().getAvailableDevices().get(0);
			}

			return preferredDevice;
		}

		public void setPreferredDevice(int preferredDevice)
		{
			this.preferredDevice = preferredDevice;
		}

		public boolean getUseOptionsString()
		{
			return useOptionsString;
		}

		public boolean getRestartLibraryAfterPhase()
		{
			return restartLibraryAfterPhase;
		}

		public void setRestartLibraryAfterPhase(boolean restartLibraryAfterPhase)
		{
			this.restartLibraryAfterPhase = restartLibraryAfterPhase;
		}

		public void setUseOptionsString(boolean useOptionsString)
		{
			if (useOptionsString && !aggregateOperations && !restartLibraryAfterPhase)
			{
				throw new RuntimeException("This option only works with aggregateOperations=true and restartLibraryAfterPhase=true");
			}

			this.useOptionsString = useOptionsString;
		}

		/**
		 * setup configuration
		 */
		public static void setUp()
		{
			RuntimeConfiguration conf = new RuntimeConfiguration();
			conf.setCalculationProvider(CalculationProvider.OPENCL);
			conf.setUseDataSharedMemory(false);
			conf.setUseWeightsSharedMemory(false);
			conf.getOpenCLConfiguration().setAggregateOperations(true);
			conf.getOpenCLConfiguration().setSynchronizeAfterOpertation(false);
			conf.getOpenCLConfiguration().setPushToDeviceBeforeOperation(false);
			conf.getOpenCLConfiguration().setFinalyzeDeviceAfterPhase(true);
			conf.getOpenCLConfiguration().setUseOptionsString(true);
			conf.getOpenCLConfiguration().setRestartLibraryAfterPhase(true);
			conf.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);

			Environment.getInstance().setRuntimeConfiguration(conf);
		}
	}

	@Override
	public String toString()
	{
		final StringBuilder sb = new StringBuilder("RuntimeConfiguration{");
		sb.append("calculationProvider=").append(calculationProvider);
		sb.append(", aparapiConfiguration=").append(aparapiConfiguration);
		sb.append(", openCLConfiguration=").append(openCLConfiguration);
		sb.append(", useDataSharedMemory=").append(useDataSharedMemory);
		sb.append(", useWeightsSharedMemory=").append(useWeightsSharedMemory);
		sb.append(", reverseSoftmaxLoss=").append(reverseSoftmaxLoss);
		sb.append('}');
		return sb.toString();
	}
}
