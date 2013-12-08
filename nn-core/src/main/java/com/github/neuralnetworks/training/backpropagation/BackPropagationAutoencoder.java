
package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.util.Properties;

/**
 * BackPropagation for autoencoders (input and target are the same)
 */
public class BackPropagationAutoencoder extends BackPropagationTrainer {

    public BackPropagationAutoencoder() {
	super();
    }

    public BackPropagationAutoencoder(Properties properties) {
	super(properties);
    }

    @Override
    protected void learnInput(TrainingInputData data) {
	super.learnInput(new AutoencoderTrainingInputData(data));
    }

    /**
     * Input and target are the same
     */
    private static class AutoencoderTrainingInputData implements TrainingInputData {

	private TrainingInputData baseInput;

	public AutoencoderTrainingInputData(TrainingInputData baseInput) {
	    this.baseInput = baseInput;
	}

	@Override
	public Matrix getInput() {
	    return baseInput.getInput();
	}

	@Override
	public Matrix getTarget() {
	    return baseInput.getInput();
	}
    }
}
