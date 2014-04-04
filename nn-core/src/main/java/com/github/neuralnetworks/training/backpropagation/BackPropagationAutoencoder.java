package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.TensorFactory;

/**
 * BackPropagation for autoencoders (input and target are the same). Supports
 * denoising autoencoders.
 */
public class BackPropagationAutoencoder extends BackPropagationTrainer<Autoencoder> {

    private static final long serialVersionUID = 1L;

    private AutoencoderTrainingInputData autoencoderTrainingInputData;

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

	private static final long serialVersionUID = 1L;

	private Tensor input;
	private Tensor target;

	@Override
	public Tensor getInput() {
	    return input;
	}

	@Override
	public Tensor getTarget() {
	    return target;
	}

	public void setInputOutput(Tensor inputOutput) {
	    this.input = inputOutput;
	    if (target == null || target.getElements().length != input.getElements().length) {
		target = TensorFactory.tensor(input.getDimensions());
	    }

	    System.arraycopy(input.getElements(), 0, target.getElements(), 0, target.getElements().length);
	}
    }
}
