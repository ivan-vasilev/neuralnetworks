package com.github.neuralnetworks.samples.iris;

import com.github.neuralnetworks.input.InputConverter;
import com.github.neuralnetworks.util.Util;

public class IrisTargetMultiNeuronOutputConverter implements InputConverter {

    private static final long serialVersionUID = 1L;

    @Override
    public void convert(Object input, float[] output) {
	Util.fillArray(output, 0);
	output[(Integer) input] = 1;
    }
}
