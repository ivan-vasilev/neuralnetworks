package com.github.neuralnetworks.util;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.util.KernelExecutionStrategy.CPUKernelExecution;
import com.github.neuralnetworks.util.KernelExecutionStrategy.DefaultKernelExecution;
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
     * is debug
     */
    private boolean debug;

    private Environment() {
	executionStrategy = new DefaultKernelExecution();
	debug = true;
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
	default:
	    this.executionStrategy = new DefaultKernelExecution();
	}
    }

    public static Environment getInstance() {
	return singleton;
    }

    public boolean isDebug() {
	return debug;
    }

    public void setDebug(boolean debug) {
	this.debug = debug;
    }
}
