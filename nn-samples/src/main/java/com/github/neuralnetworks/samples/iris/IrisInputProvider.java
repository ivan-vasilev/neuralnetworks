package com.github.neuralnetworks.samples.iris;

import java.util.Random;

import com.github.neuralnetworks.input.InputConverter;
import com.github.neuralnetworks.training.TrainingInputProviderImpl;

/**
 * Iris dataset (http://archive.ics.uci.edu/ml/datasets/Iris) with random order
 */
public class IrisInputProvider extends TrainingInputProviderImpl {

    private static final long serialVersionUID = 1L;

    private float[] target;
    private InputConverter targetConverter;
    private Random random;
    private int currentIndex;

    public IrisInputProvider(InputConverter targetConverter, boolean useRandom, boolean scale) {
	super();
	this.targetConverter = targetConverter;
	this.random = useRandom ? new Random() : null;
	this.target = new float[3];
	reset();

	if (scale) {
	    for (int i = 0; i < irisData[0].length; i++) {
		float max = irisData[0][i];
		for (int j = 0; j < irisData.length; j++) {
		    if (irisData[j][i] > max) {
			max = irisData[j][i];
		    }
		}

		for (int j = 0; j < irisData.length; j++) {
		    irisData[j][i] /= max;
		}
	    }
	}
    }

    @Override
    public void beforeSample() {
	currentIndex = random != null ? random.nextInt(150) : currentInput % 150;
    }

    @Override
    public float[] getNextInput() {
	return irisData[currentIndex];
    }

    @Override
    public float[] getNextTarget() {
	if (currentIndex < 50) {
	    targetConverter.convert(0, target);
	} else if (currentIndex < 100) {
	    targetConverter.convert(1, target);
	} else {
	    targetConverter.convert(2, target);
	}

	return target;
    }

    @Override
    public int getInputSize() {
	return irisData.length;
    }

    private float[][] irisData = {
	{ 5.1f,3.5f,1.4f,0.2f },
	{ 4.9f,3.0f,1.4f,0.2f },
	{ 4.7f,3.2f,1.3f,0.2f },
	{ 4.6f,3.1f,1.5f,0.2f },
	{ 5.0f,3.6f,1.4f,0.2f },
	{ 5.4f,3.9f,1.7f,0.4f },
	{ 4.6f,3.4f,1.4f,0.3f },
	{ 5.0f,3.4f,1.5f,0.2f },
	{ 4.4f,2.9f,1.4f,0.2f },
	{ 4.9f,3.1f,1.5f,0.1f },
	{ 5.4f,3.7f,1.5f,0.2f },
	{ 4.8f,3.4f,1.6f,0.2f },
	{ 4.8f,3.0f,1.4f,0.1f },
	{ 4.3f,3.0f,1.1f,0.1f },
	{ 5.8f,4.0f,1.2f,0.2f },
	{ 5.7f,4.4f,1.5f,0.4f },
	{ 5.4f,3.9f,1.3f,0.4f },
	{ 5.1f,3.5f,1.4f,0.3f },
	{ 5.7f,3.8f,1.7f,0.3f },
	{ 5.1f,3.8f,1.5f,0.3f },
	{ 5.4f,3.4f,1.7f,0.2f },
	{ 5.1f,3.7f,1.5f,0.4f },
	{ 4.6f,3.6f,1.0f,0.2f },
	{ 5.1f,3.3f,1.7f,0.5f },
	{ 4.8f,3.4f,1.9f,0.2f },
	{ 5.0f,3.0f,1.6f,0.2f },
	{ 5.0f,3.4f,1.6f,0.4f },
	{ 5.2f,3.5f,1.5f,0.2f },
	{ 5.2f,3.4f,1.4f,0.2f },
	{ 4.7f,3.2f,1.6f,0.2f },
	{ 4.8f,3.1f,1.6f,0.2f },
	{ 5.4f,3.4f,1.5f,0.4f },
	{ 5.2f,4.1f,1.5f,0.1f },
	{ 5.5f,4.2f,1.4f,0.2f },
	{ 4.9f,3.1f,1.5f,0.1f },
	{ 5.0f,3.2f,1.2f,0.2f },
	{ 5.5f,3.5f,1.3f,0.2f },
	{ 4.9f,3.1f,1.5f,0.1f },
	{ 4.4f,3.0f,1.3f,0.2f },
	{ 5.1f,3.4f,1.5f,0.2f },
	{ 5.0f,3.5f,1.3f,0.3f },
	{ 4.5f,2.3f,1.3f,0.3f },
	{ 4.4f,3.2f,1.3f,0.2f },
	{ 5.0f,3.5f,1.6f,0.6f },
	{ 5.1f,3.8f,1.9f,0.4f },
	{ 4.8f,3.0f,1.4f,0.3f },
	{ 5.1f,3.8f,1.6f,0.2f },
	{ 4.6f,3.2f,1.4f,0.2f },
	{ 5.3f,3.7f,1.5f,0.2f },
	{ 5.0f,3.3f,1.4f,0.2f },
	{ 7.0f,3.2f,4.7f,1.4f },
	{ 6.4f,3.2f,4.5f,1.5f },
	{ 6.9f,3.1f,4.9f,1.5f },
	{ 5.5f,2.3f,4.0f,1.3f },
	{ 6.5f,2.8f,4.6f,1.5f },
	{ 5.7f,2.8f,4.5f,1.3f },
	{ 6.3f,3.3f,4.7f,1.6f },
	{ 4.9f,2.4f,3.3f,1.0f },
	{ 6.6f,2.9f,4.6f,1.3f },
	{ 5.2f,2.7f,3.9f,1.4f },
	{ 5.0f,2.0f,3.5f,1.0f },
	{ 5.9f,3.0f,4.2f,1.5f },
	{ 6.0f,2.2f,4.0f,1.0f },
	{ 6.1f,2.9f,4.7f,1.4f },
	{ 5.6f,2.9f,3.6f,1.3f },
	{ 6.7f,3.1f,4.4f,1.4f },
	{ 5.6f,3.0f,4.5f,1.5f },
	{ 5.8f,2.7f,4.1f,1.0f },
	{ 6.2f,2.2f,4.5f,1.5f },
	{ 5.6f,2.5f,3.9f,1.1f },
	{ 5.9f,3.2f,4.8f,1.8f },
	{ 6.1f,2.8f,4.0f,1.3f },
	{ 6.3f,2.5f,4.9f,1.5f },
	{ 6.1f,2.8f,4.7f,1.2f },
	{ 6.4f,2.9f,4.3f,1.3f },
	{ 6.6f,3.0f,4.4f,1.4f },
	{ 6.8f,2.8f,4.8f,1.4f },
	{ 6.7f,3.0f,5.0f,1.7f },
	{ 6.0f,2.9f,4.5f,1.5f },
	{ 5.7f,2.6f,3.5f,1.0f },
	{ 5.5f,2.4f,3.8f,1.1f },
	{ 5.5f,2.4f,3.7f,1.0f },
	{ 5.8f,2.7f,3.9f,1.2f },
	{ 6.0f,2.7f,5.1f,1.6f },
	{ 5.4f,3.0f,4.5f,1.5f },
	{ 6.0f,3.4f,4.5f,1.6f },
	{ 6.7f,3.1f,4.7f,1.5f },
	{ 6.3f,2.3f,4.4f,1.3f },
	{ 5.6f,3.0f,4.1f,1.3f },
	{ 5.5f,2.5f,4.0f,1.3f },
	{ 5.5f,2.6f,4.4f,1.2f },
	{ 6.1f,3.0f,4.6f,1.4f },
	{ 5.8f,2.6f,4.0f,1.2f },
	{ 5.0f,2.3f,3.3f,1.0f },
	{ 5.6f,2.7f,4.2f,1.3f },
	{ 5.7f,3.0f,4.2f,1.2f },
	{ 5.7f,2.9f,4.2f,1.3f },
	{ 6.2f,2.9f,4.3f,1.3f },
	{ 5.1f,2.5f,3.0f,1.1f },
	{ 5.7f,2.8f,4.1f,1.3f },
	{ 6.3f,3.3f,6.0f,2.5f },
	{ 5.8f,2.7f,5.1f,1.9f },
	{ 7.1f,3.0f,5.9f,2.1f },
	{ 6.3f,2.9f,5.6f,1.8f },
	{ 6.5f,3.0f,5.8f,2.2f },
	{ 7.6f,3.0f,6.6f,2.1f },
	{ 4.9f,2.5f,4.5f,1.7f },
	{ 7.3f,2.9f,6.3f,1.8f },
	{ 6.7f,2.5f,5.8f,1.8f },
	{ 7.2f,3.6f,6.1f,2.5f },
	{ 6.5f,3.2f,5.1f,2.0f },
	{ 6.4f,2.7f,5.3f,1.9f },
	{ 6.8f,3.0f,5.5f,2.1f },
	{ 5.7f,2.5f,5.0f,2.0f },
	{ 5.8f,2.8f,5.1f,2.4f },
	{ 6.4f,3.2f,5.3f,2.3f },
	{ 6.5f,3.0f,5.5f,1.8f },
	{ 7.7f,3.8f,6.7f,2.2f },
	{ 7.7f,2.6f,6.9f,2.3f },
	{ 6.0f,2.2f,5.0f,1.5f },
	{ 6.9f,3.2f,5.7f,2.3f },
	{ 5.6f,2.8f,4.9f,2.0f },
	{ 7.7f,2.8f,6.7f,2.0f },
	{ 6.3f,2.7f,4.9f,1.8f },
	{ 6.7f,3.3f,5.7f,2.1f },
	{ 7.2f,3.2f,6.0f,1.8f },
	{ 6.2f,2.8f,4.8f,1.8f },
	{ 6.1f,3.0f,4.9f,1.8f },
	{ 6.4f,2.8f,5.6f,2.1f },
	{ 7.2f,3.0f,5.8f,1.6f },
	{ 7.4f,2.8f,6.1f,1.9f },
	{ 7.9f,3.8f,6.4f,2.0f },
	{ 6.4f,2.8f,5.6f,2.2f },
	{ 6.3f,2.8f,5.1f,1.5f },
	{ 6.1f,2.6f,5.6f,1.4f },
	{ 7.7f,3.0f,6.1f,2.3f },
	{ 6.3f,3.4f,5.6f,2.4f },
	{ 6.4f,3.1f,5.5f,1.8f },
	{ 6.0f,3.0f,4.8f,1.8f },
	{ 6.9f,3.1f,5.4f,2.1f },
	{ 6.7f,3.1f,5.6f,2.4f },
	{ 6.9f,3.1f,5.1f,2.3f },
	{ 5.8f,2.7f,5.1f,1.9f },
	{ 6.8f,3.2f,5.9f,2.3f },
	{ 6.7f,3.3f,5.7f,2.5f },
	{ 6.7f,3.0f,5.2f,2.3f },
	{ 6.3f,2.5f,5.0f,1.9f },
	{ 6.5f,3.0f,5.2f,2.0f },
	{ 6.2f,3.4f,5.4f,2.3f },
	{ 5.9f,3.0f,5.1f,1.8f },
    };
}