package com.github.neuralnetworks.util;

import com.aparapi.Kernel;
import com.aparapi.Kernel.EXECUTION_MODE;
import com.aparapi.Range;

/**
 * Implementations provide execution mode for Aparapi kernel
 */
public interface KernelExecutionStrategy {

    public void execute(Kernel kernel, int range);

    public static class DefaultKernelExecution implements KernelExecutionStrategy {

	@Override
	public void execute(Kernel kernel, int range) {
	    if (range <= Runtime.getRuntime().availableProcessors() * 600) {
		kernel.setExecutionMode(EXECUTION_MODE.CPU);
		kernel.execute(range);
	    } else {
		kernel.setExecutionMode(EXECUTION_MODE.GPU);
		kernel.execute(range);
	    }
	}
    }

    public static class JTPKernelExecution implements KernelExecutionStrategy {

	@Override
	public void execute(Kernel kernel, int range) {
	    kernel.setExecutionMode(EXECUTION_MODE.JTP);
	    kernel.execute(range);
	}
    }

    public static class GPUKernelExecution implements KernelExecutionStrategy {

	@Override
	public void execute(Kernel kernel, int range) {
	    kernel.setExecutionMode(EXECUTION_MODE.GPU);
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

    public static class CPUKernelExecution implements KernelExecutionStrategy {

	@Override
	public void execute(Kernel kernel, int range) {
	    kernel.setExecutionMode(EXECUTION_MODE.CPU);
	    kernel.execute(range);
	}
    }
}
