package com.github.neuralnetworks.training.backpropagation;

import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiNoise;
import com.github.neuralnetworks.training.TrainingInputData;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.TrainingInputProviderImpl;
import com.github.neuralnetworks.util.Constants;
import com.github.neuralnetworks.util.Properties;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.TensorFactory;

/**
 * BackPropagation for autoencoders (input and target are the same). Supports
 * denoising autoencoders.
 */
public class BackPropagationAutoencoder extends BackPropagationTrainer<Autoencoder> {

    private static final long serialVersionUID = 1L;

    public BackPropagationAutoencoder(Properties properties) {
	super(properties);
	setTrainingInputProvider(new AutoencoderTrainingInputrovider(getTrainingInputProvider(), getCorruptionRate()));
    }

    public Float getCorruptionRate() {
	return getProperties().getParameter(Constants.CORRUPTION_LEVEL);
    }

    public void setCorruptionRate(Float corruptionRate) {
	getProperties().setParameter(Constants.CORRUPTION_LEVEL, corruptionRate);
    }

    private static class AutoencoderTrainingInputrovider extends TrainingInputProviderImpl {

	private static final long serialVersionUID = 1L;

	private TrainingInputProvider base;
	private Float corruptionRate;
	private AparapiNoise noise;
	private Tensor noiseTensor;

	public AutoencoderTrainingInputrovider(TrainingInputProvider base, Float corruptionRate) {
	    super();
	    this.base = base;
	    this.corruptionRate = corruptionRate;
	}

	@Override
	public int getInputSize() {
	    return base.getInputSize();
	}

	@Override
	public float[] getNextInput() {
	    return base.getNextInput();
	}

	@Override
	public float[] getNextTarget() {
	    float[] result = base.getNextInput();
	    if (corruptionRate != null && corruptionRate > 0) {
		if (noise == null) {
		    noiseTensor = TensorFactory.tensor(base.getNextInput().length);
		    noise = new AparapiNoise(noiseTensor, base.getNextInput().length, corruptionRate);
		}

		System.arraycopy(result, 0, noiseTensor.getElements(), 0, result.length);
		noise.value(noiseTensor);
		result = noiseTensor.getElements();
	    }

	    return result;
	}

	@Override
	public void before(TrainingInputData ti) {
	    super.before(ti);
	    base.before(ti);
	}

	@Override
	public void after(TrainingInputData ti) {
	    super.after(ti);
	    base.after(ti);
	}

	@Override
	public void reset() {
	    super.reset();
	    base.reset();
	}
    }
}
