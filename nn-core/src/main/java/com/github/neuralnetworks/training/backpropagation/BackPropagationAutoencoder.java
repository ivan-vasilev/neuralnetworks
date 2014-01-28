package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.util.Properties;

/**
 * BackPropagation for autoencoders (input and target are the same). Supports
 * denoising autoencoders.
 */
public class BackPropagationAutoencoder extends BackPropagationTrainer<Autoencoder> {

    private AutoencoderTrainingInputData autoencoderTrainingInputData;

    public BackPropagationAutoencoder() {
	super();
    }

    public BackPropagationAutoencoder(Properties properties) {
	super(properties);
    }

    @Override
    protected void learnInput(TrainingInputData data, int batch) {
	if (autoencoderTrainingInputData == null) {
	    autoencoderTrainingInputData = new AutoencoderTrainingInputData();
	}
	autoencoderTrainingInputData.setInputOutput(data.getInput());
	super.learnInput(autoencoderTrainingInputData, batch);
    }

    /**
     * Input and target are the same
     */
    private static class AutoencoderTrainingInputData implements TrainingInputData {

	private Matrix input;
	private Matrix target;

	@Override
	public Matrix getInput() {
	    return input;
	}

	@Override
	public Matrix getTarget() {
	    return target;
	}

	public void setInputOutput(Matrix inputOutput) {
	    this.input = inputOutput;
	    if (target == null || target.getElements().length != input.getElements().length) {
		target = new Matrix(input.getRows(), input.getColumns());
	    }

	    System.arraycopy(input.getElements(), 0, target.getElements(), 0, target.getElements().length);
	}
    }
}
