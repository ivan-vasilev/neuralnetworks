package com.github.neuralnetworks.util;

import com.github.neuralnetworks.util.KernelExecutionStrategy.DefaultKernelExecution;

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

    public void setExecutionStrategy(KernelExecutionStrategy executionStrategy) {
        this.executionStrategy = executionStrategy;
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
