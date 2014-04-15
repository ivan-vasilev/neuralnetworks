package com.github.neuralnetworks.samples.mnist;

import com.github.neuralnetworks.input.InputConverter;
import com.github.neuralnetworks.util.Util;

public class MnistTargetMultiNeuronOutputConverter implements InputConverter {

    private static final long serialVersionUID = 1L;

    @Override
    public void convert(Object input, float[] output) {
	Util.fillArray(output, 0);
	output[(int) input] = 1;
    }
}
