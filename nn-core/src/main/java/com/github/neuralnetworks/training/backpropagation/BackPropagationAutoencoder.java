
package com.github.neuralnetworks.training.backpropagation;

import org.uncommons.maths.random.BinomialGenerator;

import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;

/**
 * BackPropagation for autoencoders (input and target are the same). Supports denoising autoencoders.
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
    protected void learnInput(TrainingInputData data) {
	if (autoencoderTrainingInputData == null) {
	    autoencoderTrainingInputData = new AutoencoderTrainingInputData();
	}
	autoencoderTrainingInputData.setBaseInput(data);

	Autoencoder nn = getNeuralNetwork();
	boolean useHiddenLayerAsOutput = nn.getUseHiddenLayerAsOutput();
	nn.setUseHiddenLayerAsOutput(false);

	super.learnInput(autoencoderTrainingInputData);

	nn.setUseHiddenLayerAsOutput(useHiddenLayerAsOutput);
    }


    public Float getCorruptionLevel() {
	return properties.getParameter(Constants.CORRUPTION_LEVEL);
    }

    public void setCorruptionLevel(Float corruptionLevel) {
	properties.setParameter(Constants.CORRUPTION_LEVEL, corruptionLevel);
    }

    /**
     * Input and target are the same
     */
    private class AutoencoderTrainingInputData implements TrainingInputData {

	private Matrix baseInput;
	private Matrix baseTarget;
	private BinomialGenerator binomial;

	public AutoencoderTrainingInputData() {
	    super();
	    if (getCorruptionLevel() != null && getCorruptionLevel() > 0) {
		binomial = new BinomialGenerator(1, BackPropagationAutoencoder.this.getCorruptionLevel(), getRandomInitializer().getRandom());
	    }
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
	    if (getCorruptionLevel() != null && getCorruptionLevel() > 0) {
		if (this.baseInput == null || this.baseInput.getElements().length != baseTarget.getElements().length) {
		    this.baseInput = new Matrix(baseTarget.getColumns(), baseTarget.getRows());
		}

		for (int i = 0; i < baseInput.getInput().getElements().length; i++) {
		    this.baseInput.getElements()[i] = this.baseTarget.getElements()[i] * binomial.nextValue();
		}
	    } else {
		this.baseInput = this.baseTarget;
	    }
	}
    }
}
