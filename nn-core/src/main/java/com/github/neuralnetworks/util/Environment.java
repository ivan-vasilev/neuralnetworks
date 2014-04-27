package com.github.neuralnetworks.util;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.util.KernelExecutionStrategy.CPUKernelExecution;
import com.github.neuralnetworks.util.KernelExecutionStrategy.DefaultKernelExecution;
import com.github.neuralnetworks.util.KernelExecutionStrategy.GPUKernelExecution;
import com.github.neuralnetworks.util.KernelExecutionStrategy.JTPKernelExecution;
import com.github.neuralnetworks.util.KernelExecutionStrategy.SeqKernelExecution;

/**
 * Singleton for environment variables (can be used for debugging)
 */
public class Environment {

    private static Environment singleton = new Environment();

    /**
     * Determnines whether the code will be executed on the GPU or the CPU
     */
    private KernelExecutionStrategy executionStrategy;

    /**
     * Shared memory for calculations
     */
    private boolean useDataSharedMemory;

    /**
     * Shared memory for neural network connection weights
     */
    private boolean useWeightsSharedMemory;

    private Environment() {
	executionStrategy = new DefaultKernelExecution();
	useDataSharedMemory = false;
	useWeightsSharedMemory = false;
    }

    public KernelExecutionStrategy getExecutionStrategy() {
	return executionStrategy;
    }

    public void setExecutionMode(EXECUTION_MODE executionMode) {
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

    public static Environment getInstance() {
	return singleton;
    }

    public boolean getUseDataSharedMemory() {
        return useDataSharedMemory;
    }

    public void setUseDataSharedMemory(boolean useDataSharedMemory) {
        this.useDataSharedMemory = useDataSharedMemory;
    }

    public boolean getUseWeightsSharedMemory() {
        return useWeightsSharedMemory;
    }

    public void setUseWeightsSharedMemory(boolean useWeightsSharedMemory) {
        this.useWeightsSharedMemory = useWeightsSharedMemory;
    }
}
