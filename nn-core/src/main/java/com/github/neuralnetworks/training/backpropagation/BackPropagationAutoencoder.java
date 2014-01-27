package com.github.neuralnetworks.training.backpropagation;

import java.util.HashMap;
import java.util.Map;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.util.Constants;
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
	autoencoderTrainingInputData.setBaseInput(data);
	super.learnInput(autoencoderTrainingInputData, batch);
    }
    
    public InputCorruptor getInputCorruptor() {
	return properties.getParameter(Constants.CORRUPTOR);
    }
    
    public void setInputCorruptor(InputCorruptor corruptor) {
	properties.setParameter(Constants.CORRUPTOR, corruptor);
    }

    /**
     * Input and target are the same
     */
    private class AutoencoderTrainingInputData implements TrainingInputData {

	private Matrix baseInput;
	private Matrix baseTarget;
	protected Map<Integer, float[]> randomDistributions = new HashMap<>();

	public AutoencoderTrainingInputData() {
	    super();
	}

	@Override
	public Matrix getInput() {
	    return baseInput;
	}

	@Override
	public Matrix getTarget() {
	    return baseTarget;
	}

	public void setBaseInput(TrainingInputData baseInput) {
	    this.baseTarget = baseInput.getInput();
	    if (getInputCorruptor() != null) {
		float[] randomDistribution = randomDistributions.get(baseTarget.getElements().length);
		if (randomDistribution == null) {
		    randomDistribution = new float[baseTarget.getElements().length];
		    randomDistributions.put(randomDistribution.length, randomDistribution);
		}

		if (this.baseInput == null || this.baseInput.getElements().length != baseTarget.getElements().length) {
		    this.baseInput = new Matrix(baseTarget.getColumns(), baseTarget.getRows());
		}

		getInputCorruptor().corrupt(this.baseInput.getElements());
	    } else {
		this.baseInput = this.baseTarget;
	    }
	}
    }
}
