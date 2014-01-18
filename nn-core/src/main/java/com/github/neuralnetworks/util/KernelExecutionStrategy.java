package com.github.neuralnetworks.util;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.amd.aparapi.Range;

/**
 * Implementations provide execution mode for Aparapi kernel
 */
public interface KernelExecutionStrategy {

    public void execute(Kernel kernel, int range);

    public static class DefaultKernelExecution implements KernelExecutionStrategy {

	@Override
	public void execute(Kernel kernel, int range) {
	    if (range <= Runtime.getRuntime().availableProcessors() * 5) {
		kernel.setExecutionMode(EXECUTION_MODE.CPU);
	    } else {
		kernel.setExecutionMode(EXECUTION_MODE.GPU);
	    }

	    kernel.execute(range);
	}
    }

    public static class JTPKernelExecution implements KernelExecutionStrategy {

	@Override
	public void execute(Kernel kernel, int range) {
	    kernel.setExecutionMode(EXECUTION_MODE.JTP);
	    kernel.execute(range);
	}
    }

    public static class SeqKernelExecution implements KernelExecutionStrategy {

	@Override
	public void execute(Kernel kernel, int range) {
	    kernel.setExecutionMode(EXECUTION_MODE.SEQ);
	    kernel.execute(Range.create(range, 1));
	}
    }
}
