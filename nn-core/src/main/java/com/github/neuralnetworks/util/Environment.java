package com.github.neuralnetworks.util;

import com.amd.aparapi.Kernel.EXECUTION_MODE;

/**
 * Singleton for environment variables (can be used for debugging)
 */
public class Environment {

    private static Environment singleton = new Environment();

    /**
     * Determnines whether the code will be executed on the GPU or the CPU
     */
    private EXECUTION_MODE executionMode;

    /**
     * is debug
     */
    private boolean debug;

    private Environment() {
	executionMode = EXECUTION_MODE.GPU;
	debug = true;
    }

    public EXECUTION_MODE getExecutionMode() {
        return executionMode;
    }

    public void setExecutionMode(EXECUTION_MODE executionMode) {
        this.executionMode = executionMode;
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
