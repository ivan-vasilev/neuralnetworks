package com.github.neuralnetworks.util;

import com.amd.aparapi.Kernel.EXECUTION_MODE;

/**
 * 
 * Singleton for execution mode (can be used for debugging)
 *
 */
public class AparapiExecutionMode {

    private static AparapiExecutionMode singleton = new AparapiExecutionMode();
    private EXECUTION_MODE executionMode;

    private AparapiExecutionMode() {
	executionMode = EXECUTION_MODE.GPU;
    }

    public EXECUTION_MODE getExecutionMode() {
        return executionMode;
    }

    public void setExecutionMode(EXECUTION_MODE executionMode) {
        this.executionMode = executionMode;
    }

    public static AparapiExecutionMode getInstance() {
	return singleton;
    }
}
