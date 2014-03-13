package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.HashMap;
import java.util.Map;

import com.github.neuralnetworks.training.random.XORShiftKernel;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Matrix;

/**
 * Bernoulli distribution
 */
public class BernoulliDistribution implements MatrixFunction {

    private static final long serialVersionUID = 1L;

    protected Map<Integer, BernoulliKernel> kernels = new HashMap<>();

    @Override
    public void value(Matrix inputOutput) {
	BernoulliKernel kernel = kernels.get(inputOutput.getElements().length);
	if (kernel == null) {
	    kernels.put(inputOutput.getElements().length, kernel = new BernoulliKernel(inputOutput.getElements().length));
	}

	kernel.values = inputOutput.getElements();

	Environment.getInstance().getExecutionStrategy().execute(kernel, inputOutput.getElements().length);
    }

    private static class BernoulliKernel extends XORShiftKernel {

	private float[] values;

	public BernoulliKernel(int maximumRange) {
	    super(maximumRange);
	}

	@Override
	public void run() {
	    int id = getGlobalId();
	    if (values[id] > random01()) {
		values[id] = 1;
	    } else {
		values[id] = 0;
	    }
	}
    }
}
